// frontend/app/api/upload/image/route.ts
import { NextResponse } from 'next/server'
import { cookies } from 'next/headers'

export async function POST(request: Request) {
  try {
    // Parse the incoming form data
    const formData = await request.formData()

    // Get the JWT token from cookies
    const cookieStore = cookies()
    const token = (await cookieStore).get('jwt_token')?.value
    const authHeader = token ? `Bearer ${token}` : ''

    // Forward the request to the backend
    const backendUrl = `${process.env.NEXT_PUBLIC_BACKEND_API_URL}/upload/image`
    const backendResponse = await fetch(backendUrl, {
      method: 'POST',
      headers: {
        // Let the browser set the Content-Type for form data
        Authorization: authHeader
      },
      body: formData
    })

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text()
      return NextResponse.json(
        { error: errorText },
        { status: backendResponse.status }
      )
    }

    // Return the backend response (which includes the image ID)
    const data = await backendResponse.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Error in image upload API route:', error)
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    )
  }
}
