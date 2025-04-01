import React from 'react'
import { CollapsibleMessage } from './collapsible-message'
import Image from 'next/image'

interface UserMessageProps {
  message: string
  imagePreview?: string | null
}

export function UserMessage({ message, imagePreview }: UserMessageProps) {
  // Process imagePreview if it contains [IMAGE] tags
  const processedImagePreview = imagePreview?.startsWith('[IMAGE]')
    ? `data:image/jpeg;base64,${imagePreview.replace(
        /\[IMAGE\]|\[\/IMAGE\]/g,
        ''
      )}`
    : imagePreview

  return (
    <div className="prose dark:prose-invert max-w-none">
      <p className="text-base leading-relaxed whitespace-pre-wrap">{message}</p>
      {processedImagePreview && (
        <div className="mt-2">
          <Image
            src={processedImagePreview}
            alt="Uploaded image"
            width={300}
            height={300}
            className="rounded-lg max-w-full h-auto"
            priority
          />
        </div>
      )}
    </div>
  )
}
