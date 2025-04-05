import { cookies } from 'next/headers'
import { NextResponse } from 'next/server'

type Analysis = Promise<{ sessionId: string }>

export async function GET(request: Request, { params }: { params: Analysis }) {
  const cookieStore = await cookies()
  const token = cookieStore.get('jwt_token')?.value
  const sessionId = (await params).sessionId

  if (!token) {
    return new Response('Unauthorized', { status: 401 })
  }

  try {
    const response = await fetch(
      `${process.env.NEXT_PUBLIC_BACKEND_API_URL}/chat/${sessionId}/analysis`,
      {
        method: 'GET',
        headers: {
          Authorization: `Bearer ${token}`
        }
      }
    )

    if (!response.ok) {
      const error = await response.json()
      return NextResponse.json(
        { error: error.detail },
        { status: response.status }
      )
    }

    const analysis = await response.json()
    return NextResponse.json(analysis)
  } catch (error) {
    console.error('Error fetching analysis:', error)
    return NextResponse.json(
      { error: 'Failed to fetch performance analysis' },
      { status: 500 }
    )
  }
}
