// frontend/app/api/chat/route.ts
import { cookies } from 'next/headers'
import { NextResponse } from 'next/server'

export const maxDuration = 30

export async function POST(req: Request) {
  try {
    // Destructure session_id (instead of sessionId) along with message and optional image_id
    const { message, session_id, image_id } = await req.json()
    const cookieStore = await cookies()
    const token = cookieStore.get('jwt_token')?.value

    if (!token) {
      return new Response('Unauthorized', { status: 401 })
    }

    if (!session_id) {
      return new Response('Session ID required', { status: 400 })
    }

    const response = await fetch(
      `${process.env.NEXT_PUBLIC_BACKEND_API_URL}/chat/stream_http`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          session_id: session_id, // Use session_id from payload
          message: message,
          image_id: image_id // Pass image_id if provided
        })
      }
    )

    if (!response.ok) {
      const errorText = await response.text()
      console.error('Backend error:', response.status, errorText)
      throw new Error('Network response was not ok')
    }

    return new Response(response.body, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive'
      }
    })
  } catch (error) {
    console.error('API route error:', error)
    return new Response('Error processing your request', { status: 500 })
  }
}
