import base64
import os
import json
import re
import uuid
from datetime import datetime, timedelta
from uuid import UUID as uuid_UUID
import asyncio

import boto3.session
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
import sqlalchemy
import uvicorn
from passlib.context import CryptContext
from typing import AsyncGenerator, List, Optional, Dict

# Import LLM and tools (assuming these exist in your codebase)
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import BraveSearch
from langchain_core.messages import SystemMessage, HumanMessage
from app.chat_provider.service.chat_service import ChatService
from redis.asyncio import Redis
import boto3

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_client = Redis.from_url(redis_url, decode_responses=True)


# Database Connection
def connect_tcp_socket() -> sqlalchemy.engine.base.Engine:
    """Initializes a TCP connection pool for a Cloud SQL instance of Postgres."""
    db_host = os.environ.get("INSTANCE_HOST", "some-db-ip")
    db_user = os.environ.get("DB_USER", "postgres")
    db_pass = os.environ.get("DB_PASS", "some-db-password")
    db_name = os.environ.get("DB_NAME", "postgres")
    db_port = os.environ.get("DB_PORT", "5432")

    pool = sqlalchemy.create_engine(
        sqlalchemy.engine.url.URL.create(
            drivername="postgresql+pg8000",
            username=db_user,
            password=db_pass,
            host=db_host,
            port=db_port,
            database=db_name,
        ),
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
    )
    return pool


def initialize_digitalocean_bucket():
    bucket_url = os.environ.get("DIGITAL_OCEAN_BUCKET_URL", "https://some-bucket-url")
    access_key = os.environ.get("DIGITAL_OCEAN_ACCESS_KEY", "some-access-key")
    access_key_id = os.environ.get("DIGITAL_OCEAN_ACCESS_KEY_ID", "some-access-key-id")
    session = boto3.session.Session()
    s3_client = session.client(
        "s3",
        region_name=os.environ.get("DIGITAL_OCEAN_REGION_NAME", "blr1"),
        endpoint_url=bucket_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=access_key,
    )
    return s3_client


# Async engine setup
sync_engine = connect_tcp_socket()
engine = create_async_engine(
    sqlalchemy.engine.url.URL.create(
        drivername="postgresql+asyncpg",
        username=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASS", "some-password"),
        host=os.environ.get("INSTANCE_HOST", "some-dp-ip"),
        port=os.environ.get("DB_PORT", "5432"),
        database=os.environ.get("DB_NAME", "postgres"),
    ),
    echo=True,
)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


# Database Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    summary = Column(String, nullable=True)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id"))
    sender = Column(String)
    message = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    sources = Column(JSON, nullable=True)


# New ImageUpload Model for storing image metadata
class ImageUpload(Base):
    __tablename__ = "image_uploads"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    file_name = Column(String, nullable=False)
    file_url = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)


# Create tables on startup
@app.on_event("startup")
async def startup_event():
    Base.metadata.create_all(sync_engine)


# Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


# JWT Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "some-secrey-key")
ALGORITHM = "HS256"
JWT_EXPIRE_TIME = int(os.environ.get("JWT_EXPIRE_TIME", 15))


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=JWT_EXPIRE_TIME))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# Database Dependency
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# Authentication Dependency
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    user = result.scalars().first()
    if not user:
        raise credentials_exception
    return user


# Pydantic Models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


# Extend ChatInput with optional image_id parameter
class ChatInput(BaseModel):
    session_id: str  # now a UUID in string format
    message: str
    image_id: Optional[str] = None


class ChatResponse(BaseModel):
    message: str
    sources: List = []


# Safety Settings for LLM
safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

# Environment Variables for LLM and search tools
GEMINI_API_KEY = os.environ.get("GOOGLE_GEMINI_API_KEY")
BRAVE_API_KEY = os.environ.get("BRAVE_SEARCH_API_KEY")
GOOGLE_SEARCH_API_KEY = os.environ.get("GOOGLE_SEARCH_API_KEY")
TAVILY_SEARCH_API_KEY = os.environ.get("TAVILY_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_SEACH_ENGINE_ID")

# Initialize LLM and Tools
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-pro-exp-02-05",
    api_key=GEMINI_API_KEY,
    safety_settings=safety_settings,
)
search = GoogleSearchAPIWrapper(
    google_api_key=GOOGLE_SEARCH_API_KEY, google_cse_id=GOOGLE_CSE_ID
)
tavily_tool = TavilySearchResults(tavily_api_key=TAVILY_SEARCH_API_KEY)
google_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", google_api_key=GEMINI_API_KEY
)
brave_search = BraveSearch.from_api_key(
    api_key=BRAVE_API_KEY, search_kwargs={"count": 3}
)


# ChatServiceManager
class ChatServiceManager:
    def __init__(self):
        # Dictionary to hold ChatService instances per session
        self.chat_services = {}
        self.semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent requests

    def get_chat_service(self, session_id: str):
        # If there's no ChatService for this session, create one
        if session_id not in self.chat_services:
            self.chat_services[session_id] = ChatService(
                llm=llm,
                google_search_wrapper=search,
                google_embedings=google_embeddings,
                tavily_tool=tavily_tool,
                brave_search=brave_search,
            )
        return self.chat_services[session_id]

    async def process_message(
        self, session_id: str, message: str, chat_history: list
    ) -> ChatResponse:
        try:
            async with self.semaphore:
                # Get the ChatService instance specific to this session
                chat_service = self.get_chat_service(session_id)
                # Process input with the (possibly augmented) message
                result = chat_service.process_input(message)
                if asyncio.iscoroutine(result):
                    response = await result
                else:
                    response = result
                return ChatResponse(message=response, sources=[])
        except Exception as e:
            return ChatResponse(message=f"An error occurred: {str(e)}", sources=[])

    async def chat_summary(self, session_id: str, chat_history: list) -> str:
        try:
            async with self.semaphore:
                chat_service = self.get_chat_service(session_id)
                summary = await chat_service.chat_summary(chat_history)
                return summary
        except Exception as e:
            print(f"Error in ChatServiceManager: {str(e)}")
            return "Error generating summary"

    async def stream_message(
        self, session_id: str, message: str, chat_history: list
    ) -> AsyncGenerator[str, None]:
        try:
            async with self.semaphore:
                chat_service = self.get_chat_service(session_id)
                async for token in chat_service.stream_input(message):
                    if token and token.strip():
                        await asyncio.sleep(0.01)
                        yield token
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"Streaming error in ChatServiceManager: {str(e)}")
            yield f"Error: {str(e)}"


async def get_chat_history(
    session_id: str, db: AsyncSession, force_db: bool = False
) -> List[dict]:
    """
    Retrieve chat history for a session, using Redis cache when available.
    """
    cache_key = f"chat_history:{session_id}"

    # Try Redis first unless forced to use database
    if not force_db:
        history_json = await redis_client.get(cache_key)
        if history_json:
            return json.loads(history_json)

    # Fetch from database
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == uuid_UUID(session_id))
        .order_by(ChatMessage.timestamp)
    )
    result = await db.execute(stmt)
    chat_history = [
        {
            "sender": msg.sender,
            "message": msg.message,
            "timestamp": msg.timestamp.isoformat(),
        }
        for msg in result.scalars().all()
    ]

    # Cache in Redis with a TTL of 5 minutes (300 seconds)
    await redis_client.set(cache_key, json.dumps(chat_history), ex=300)
    return chat_history


# Helper function to fetch and encode image from S3 based on image_id
async def get_image_base64_from_db(image_id: str, db: AsyncSession) -> str:
    # Query the image record from the database
    stmt = select(ImageUpload).where(ImageUpload.id == uuid.UUID(image_id))
    result = await db.execute(stmt)
    image_record = result.scalars().first()
    if not image_record:
        raise HTTPException(status_code=404, detail="Image not found")

    # Extract the S3 key from the stored file URL
    file_url = image_record.file_url
    bucket_url = os.environ.get("DIGITAL_OCEAN_BUCKET_URL", "https://some-bucket-url")
    if not file_url.startswith(bucket_url):
        raise HTTPException(status_code=500, detail="Invalid image file URL")
    # Assume the S3 key is the part after the bucket URL and a slash
    s3_key = file_url[len(bucket_url) + 1 :]

    s3_client = initialize_digitalocean_bucket()
    bucket_name = os.environ.get("DIGITAL_OCEAN_BUCKET_NAME", "pragati-bucket")

    def fetch_image():
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        image_data = response["Body"].read()
        return base64.b64encode(image_data).decode("utf-8")

    base64_image = await asyncio.to_thread(fetch_image)
    return base64_image


chat_service_manager = ChatServiceManager()

# Endpoints


@app.post("/register")
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    stmt = select(User).where(
        (User.username == user.username) | (User.email == user.email)
    )
    result = await db.execute(stmt)
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Username or email already exists")
    hashed_password = hash_password(user.password)
    new_user = User(
        username=user.username, email=user.email, hashed_password=hashed_password
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return {"message": "User created successfully"}


@app.post("/login")
async def login(user: UserLogin, db: AsyncSession = Depends(get_db)):
    stmt = select(User).where(User.username == user.username)
    result = await db.execute(stmt)
    db_user = result.scalars().first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": db_user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/upload/image")
async def upload_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    bucket_name = os.environ.get("DIGITAL_OCEAN_BUCKET_NAME", "pragati-bucket")
    s3_client = initialize_digitalocean_bucket()

    # Generate a unique filename using UUID while preserving the original extension
    original_filename = file.filename
    file_extension = os.path.splitext(original_filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    s3_file_key = f"uploads/{unique_filename}"

    try:
        # Upload the file object asynchronously to S3 using a thread executor
        await asyncio.to_thread(
            s3_client.upload_fileobj, file.file, bucket_name, s3_file_key
        )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Failed to upload image")

    # Construct the file URL (assumes bucket URL is accessible)
    bucket_url = os.environ.get("DIGITAL_OCEAN_BUCKET_URL", "https://some-bucket-url")
    file_url = f"{bucket_url}/{s3_file_key}"

    # Create a new image record in the database
    new_image = ImageUpload(
        user_id=current_user.id,
        file_name=original_filename,
        file_url=file_url,
    )
    db.add(new_image)
    await db.commit()
    await db.refresh(new_image)

    return {
        "image_id": str(new_image.id),
        "file_name": original_filename,
        "file_url": file_url,
    }


@app.post("/sessions")
async def create_session(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    new_session = ChatSession(user_id=current_user.id)
    db.add(new_session)
    await db.commit()
    await db.refresh(new_session)
    # Return the UUID as a string
    return {"session_id": str(new_session.id)}


@app.get("/sessions")
async def list_sessions(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    stmt = select(ChatSession).where(ChatSession.user_id == current_user.id)
    result = await db.execute(stmt)
    sessions = result.scalars().all()
    return [
        {
            "session_id": str(session.id),
            "created_at": session.created_at.isoformat(),
            "summary": session.summary,
        }
        for session in sessions
    ]


@app.post("/chat")
async def send_message(
    input_data: ChatInput,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        session_uuid = uuid_UUID(input_data.session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session id format")

    stmt = select(ChatSession).where(
        ChatSession.id == session_uuid, ChatSession.user_id == current_user.id
    )
    result = await db.execute(stmt)
    session = result.scalars().first()
    if not session:
        raise HTTPException(
            status_code=404, detail="Session not found or not authorized"
        )

    # If an image_id is provided, fetch its base64 representation and append it to the message
    augmented_message = input_data.message
    if input_data.image_id:
        image_base64 = await get_image_base64_from_db(input_data.image_id, db)
        # Wrap the base64 string with markers so the ChatService can detect and parse it
        augmented_message = f"{augmented_message}\n[IMAGE]{image_base64}[/IMAGE]"

    # Save user's message
    user_message = ChatMessage(
        session_id=session.id,
        sender="user",
        message=augmented_message,
        timestamp=datetime.utcnow(),
    )
    db.add(user_message)
    await db.commit()

    # Retrieve chat history with Redis caching
    chat_history = await get_chat_history(input_data.session_id, db)

    # Process the message with the (possibly augmented) content
    response = await chat_service_manager.process_message(
        input_data.session_id, augmented_message, chat_history
    )

    # Save bot's reply
    bot_message = ChatMessage(
        session_id=session.id,
        sender="bot",
        message=response.message,
        timestamp=datetime.utcnow(),
        sources=response.sources,
    )
    db.add(bot_message)
    await db.commit()

    # Update Redis cache with the latest history
    await get_chat_history(input_data.session_id, db, force_db=True)

    return response


@app.post("/chat/stream_http")
async def stream_chat(
    input_data: ChatInput,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        session_uuid = uuid_UUID(input_data.session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session id format")

    stmt = select(ChatSession).where(
        ChatSession.id == session_uuid, ChatSession.user_id == current_user.id
    )
    result = await db.execute(stmt)
    session = result.scalars().first()
    if not session:
        raise HTTPException(
            status_code=404, detail="Session not found or not authorized"
        )

    # If an image_id is provided, fetch its base64 representation and append it to the message
    augmented_message = input_data.message
    if input_data.image_id:
        image_base64 = await get_image_base64_from_db(input_data.image_id, db)
        augmented_message = f"{augmented_message}\n[IMAGE]{image_base64}[/IMAGE]"

    # Save user's message
    user_message = ChatMessage(
        session_id=session.id,
        sender="user",
        message=augmented_message,
        timestamp=datetime.utcnow(),
    )
    db.add(user_message)
    await db.commit()

    # Retrieve chat history with Redis caching
    chat_history = await get_chat_history(input_data.session_id, db)

    async def stream_generator():
        full_response = ""
        sources = []
        try:
            yield 'data: {"type":"heartbeat"}\n\n'

            async for token in chat_service_manager.stream_message(
                input_data.session_id, augmented_message, chat_history
            ):
                if token:
                    parts = re.split(r"(\s+)", token)
                    for part in parts:
                        if part:
                            full_response += part
                            yield f'data: {{"type":"token","content":{json.dumps(part)}}}\n\n'
                            await asyncio.sleep(0.01)

            # Save bot's full reply
            bot_message = ChatMessage(
                session_id=session.id,
                sender="bot",
                message=full_response,
                timestamp=datetime.utcnow(),
                sources=sources,
            )
            db.add(bot_message)
            await db.commit()

            # Update Redis cache
            await get_chat_history(input_data.session_id, db, force_db=True)

            # Generate and update summary after we have the complete response
            try:
                print(f"\n=== Starting Summary Generation for Session {input_data.session_id} ===")
                updated_history = await get_chat_history(input_data.session_id, db, force_db=True)
                print(f"Retrieved updated chat history with {len(updated_history)} messages")
                
                # Format the chat history properly for the LLM
                formatted_history = []
                for msg in updated_history:
                    if msg["sender"] == "user":
                        formatted_history.append({
                            "role": "user",
                            "content": msg["message"]
                        })
                    else:
                        formatted_history.append({
                            "role": "assistant",
                            "content": msg["message"]
                        })
                
                summary = await chat_service_manager.chat_summary(input_data.session_id, formatted_history)
                print(f"Generated summary: {summary}")
                
                # Update the session summary
                print(f"Updating session {input_data.session_id} with new summary")
                
                # Get a fresh reference to the session
                stmt = select(ChatSession).where(ChatSession.id == session_uuid)
                result = await db.execute(stmt)
                current_session = result.scalars().first()
                
                if current_session:
                    current_session.summary = summary
                    await db.commit()
                    await db.refresh(current_session)
                    print(f"Successfully updated session summary")
                else:
                    print(f"Failed to find session {input_data.session_id} for update")
                
                print("=== Summary Update Complete ===\n")
            except Exception as e:
                print(f"\n=== Error in Summary Update ===")
                print(f"Session ID: {input_data.session_id}")
                print(f"Error details: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                print("=== Summary Update Failed ===\n")
                # Don't fail the whole request if summary update fails

            yield 'data: {"type":"complete","finishReason":"stop"}\n\n'

        except asyncio.CancelledError:
            yield 'data: {"type":"error","finishReason":"cancelled"}\n\n'
        except Exception as e:
            print(f"Streaming error: {str(e)}")
            yield f'data: {{"type":"error","finishReason":"error","error":{json.dumps(str(e))}}}\n\n'

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/chat/history")
async def get_chat_history_endpoint(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        session_uuid = uuid_UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session id format")

    stmt = select(ChatSession).where(
        ChatSession.id == session_uuid, ChatSession.user_id == current_user.id
    )
    result = await db.execute(stmt)
    session = result.scalars().first()
    if not session:
        raise HTTPException(
            status_code=404, detail="Session not found or not authorized"
        )

    # Get chat history with Redis caching
    chat_history = await get_chat_history(session_id, db)

    # Format for response
    formatted_history = [
        {
            "id": str(
                uuid.uuid4()
            ),  # Generate a temporary ID since we fetch from cache
            "role": "user" if msg["sender"] == "user" else "assistant",
            "content": msg["message"],
            "timestamp": msg["timestamp"],
            "sources": msg.get("sources", []),
        }
        for msg in chat_history
    ]

    return formatted_history


@app.on_event("shutdown")
async def shutdown_event():
    await redis_client.close()


# Add these new models after the existing ones
class PerformanceMetrics(BaseModel):
    overall_rating: float
    learning_speed: float
    engagement_level: float
    strengths: list[str]
    areas_to_improve: list[str]
    recommendations: list[str]
    next_steps: list[str]

class ChatAnalysisResponse(BaseModel):
    metrics: PerformanceMetrics

@app.get("/chat/{session_id}/analysis")
async def analyze_chat_performance(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        session_uuid = uuid_UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session id format")

    # Verify session ownership
    stmt = select(ChatSession).where(
        ChatSession.id == session_uuid, ChatSession.user_id == current_user.id
    )
    result = await db.execute(stmt)
    session = result.scalars().first()
    if not session:
        raise HTTPException(
            status_code=404, detail="Session not found or not authorized"
        )

    # Get chat history
    chat_history = await get_chat_history(session_id, db)

    # Format chat history for analysis
    formatted_history = []
    for msg in chat_history:
        if msg["sender"] == "user":
            formatted_history.append({
                "role": "user",
                "content": msg["message"]
            })
        else:
            formatted_history.append({
                "role": "assistant",
                "content": msg["message"]
            })

    try:
        # Create a prompt for the LLM to analyze the conversation
        analysis_prompt = f"""Please analyze this conversation and provide performance metrics in JSON format. Your response must be a valid JSON object with the following structure:

{{
    "metrics": {{
        "overall_rating": <float between 0-100>,
        "learning_speed": <float between 0-100>,
        "engagement_level": <float between 0-100>,
        "strengths": [<list of 3-5 strengths>],
        "areas_to_improve": [<list of 3-5 areas>],
        "recommendations": [<list of 3-5 specific recommendations>],
        "next_steps": [<list of 3-5 concrete next steps>]
    }}
}}

Important:
1. Your response must be ONLY the JSON object, no additional text
2. All values must be properly formatted (numbers for ratings, arrays for lists)
3. The response must be valid JSON that can be parsed

Conversation:
{json.dumps(formatted_history, indent=2)}

Analysis:"""

        # Get the ChatService instance for this session
        chat_service = chat_service_manager.get_chat_service(session_id)
        
        # Create messages for the LLM
        messages = [
            SystemMessage(content="You are an expert at analyzing learning conversations and providing detailed performance metrics. Your responses must be in valid JSON format only."),
            HumanMessage(content=analysis_prompt)
        ]
        
        # Get the response from the LLM
        response = await chat_service.llm.ainvoke(messages)
        
        # Try to parse as JSON first
        try:
            # Clean the response content to ensure it's valid JSON
            content = response.content.strip()
            # Remove any markdown code block markers if present
            content = content.replace('```json', '').replace('```', '').strip()
            analysis_data = json.loads(content)
            
            # Validate the response structure
            if not isinstance(analysis_data, dict) or 'metrics' not in analysis_data:
                raise ValueError("Invalid response structure")
                
            metrics = analysis_data['metrics']
            if not all(key in metrics for key in ['overall_rating', 'learning_speed', 'engagement_level', 'strengths', 'areas_to_improve', 'recommendations', 'next_steps']):
                raise ValueError("Missing required metrics")
                
            return ChatAnalysisResponse(metrics=metrics)
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing JSON response: {str(e)}")
            print(f"Raw response: {response.content}")
            # If JSON parsing fails, return a basic response
            return ChatAnalysisResponse(metrics=PerformanceMetrics(
                overall_rating=50.0,
                learning_speed=50.0,
                engagement_level=50.0,
                strengths=["Unable to analyze strengths"],
                areas_to_improve=["Unable to analyze areas for improvement"],
                recommendations=["Unable to generate recommendations"],
                next_steps=["Unable to suggest next steps"]
            ))
            
    except Exception as e:
        print(f"Error analyzing chat performance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error analyzing chat performance"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
