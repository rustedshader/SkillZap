'use client'
import { useEffect, useState, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { toast } from 'sonner'
import {
  Loader2,
  TrendingUp,
  Target,
  Lightbulb,
  Clock,
  Activity
} from 'lucide-react'

interface PerformanceMetrics {
  overall_rating: number
  learning_speed: number
  engagement_level: number
  strengths: string[]
  areas_to_improve: string[]
  recommendations: string[]
  next_steps: string[]
}

interface ChatAnalysis {
  metrics: PerformanceMetrics
}

export function PerformanceAnalysis({ sessionId }: { sessionId: string }) {
  const [analysis, setAnalysis] = useState<ChatAnalysis | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const mountedRef = useRef(false)

  useEffect(() => {
    async function fetchAnalysis() {
      if (mountedRef.current) return
      mountedRef.current = true

      try {
        setIsLoading(true)
        const res = await fetch(`/api/chat/${sessionId}/analysis`)
        if (res.ok) {
          const data = await res.json()
          setAnalysis(data)
        } else {
          const error = await res.json()
          toast.error(error.error || 'Failed to load performance analysis')
        }
      } catch (error) {
        console.error(error)
        toast.error('Error loading performance analysis')
      } finally {
        setIsLoading(false)
      }
    }
    fetchAnalysis()
  }, [sessionId])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="size-8 animate-spin text-gray-400" />
      </div>
    )
  }

  if (!analysis) {
    return null
  }

  const { metrics } = analysis

  return (
    <div className="space-y-6 p-6 bg-white rounded-lg shadow-sm">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Performance Analysis</h2>
        <div className="flex items-center gap-2">
          <TrendingUp className="size-5 text-green-500" />
          <span className="text-2xl font-bold text-green-500">
            {metrics.overall_rating}%
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <Target className="size-5 text-blue-500" />
            <h3 className="font-semibold">Overall Rating</h3>
          </div>
          <div className="text-3xl font-bold text-blue-500">
            {metrics.overall_rating.toFixed(1)}%
          </div>
        </div>

        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="size-5 text-purple-500" />
            <h3 className="font-semibold">Learning Speed</h3>
          </div>
          <div className="text-xl font-semibold text-purple-500">
            {metrics.learning_speed}%
          </div>
        </div>

        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="size-5 text-orange-500" />
            <h3 className="font-semibold">Engagement Level</h3>
          </div>
          <div className="text-xl font-semibold text-orange-500">
            {metrics.engagement_level}%
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 className="text-lg font-semibold mb-3">Strengths</h3>
          <ul className="space-y-2">
            {metrics.strengths.map((strength, index) => (
              <li key={index} className="flex items-start gap-2">
                <span className="text-green-500">•</span>
                <span>{strength}</span>
              </li>
            ))}
          </ul>
        </div>

        <div>
          <h3 className="text-lg font-semibold mb-3">Areas to Improve</h3>
          <ul className="space-y-2">
            {metrics.areas_to_improve.map((area, index) => (
              <li key={index} className="flex items-start gap-2">
                <span className="text-red-500">•</span>
                <span>{area}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-3">Recommendations</h3>
        <ul className="space-y-2">
          {metrics.recommendations.map((rec, index) => (
            <li key={index} className="flex items-start gap-2">
              <Lightbulb className="size-5 text-yellow-500 mt-1" />
              <span>{rec}</span>
            </li>
          ))}
        </ul>
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-3">Next Steps</h3>
        <ul className="space-y-2">
          {metrics.next_steps.map((step, index) => (
            <li key={index} className="flex items-start gap-2">
              <span className="text-blue-500">→</span>
              <span>{step}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
