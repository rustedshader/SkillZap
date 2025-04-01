import React from 'react'
import { CollapsibleMessage } from './collapsible-message'
import Image from 'next/image'

interface UserMessageProps {
  message: string
  imagePreview?: string | null
}

export function UserMessage({ message, imagePreview }: UserMessageProps) {
  return (
    <div className="prose dark:prose-invert max-w-none">
      <p className="text-base leading-relaxed whitespace-pre-wrap">{message}</p>
      {imagePreview && (
        <div className="mt-2">
          <Image
            src={imagePreview}
            alt="Uploaded image"
            width={300}
            height={300}
            className="rounded-lg max-w-full h-auto"
          />
        </div>
      )}
    </div>
  )
}
