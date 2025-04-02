// Inside ChatPanel component in /frontend/components/chat-panel.tsx
import { ArrowUp, Image, Square, Loader2, BarChart3 } from 'lucide-react'
import { useRef, useState } from 'react'
import { Button } from './ui/button'
import Textarea from 'react-textarea-autosize'
import { toast } from 'sonner'

interface ChatPanelProps {
  input: string
  handleInputChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void
  handleSubmit: (e: React.FormEvent<HTMLFormElement>) => void
  isLoading: boolean
  messages: any[]
  setMessages: (messages: any[]) => void
  stop: () => void
  append: (message: any) => void
  setImageId: (id: string | null) => void
  imagePreview: string | null
  setImagePreview: (url: string | null) => void
  onShowAnalysis: () => void
}

export function ChatPanel({
  input,
  handleInputChange,
  handleSubmit,
  isLoading,
  messages,
  setMessages,
  stop,
  append,
  setImageId,
  imagePreview,
  setImagePreview,
  onShowAnalysis
}: ChatPanelProps) {
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isComposing, setIsComposing] = useState(false)
  const [isUploading, setIsUploading] = useState(false)

  const handleImageButtonClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      // Create a preview URL to display in the UI
      const previewUrl = URL.createObjectURL(file)
      setImagePreview(previewUrl)
      setIsUploading(true)

      // Prepare the file upload using FormData
      const formData = new FormData()
      formData.append('file', file)

      try {
        const response = await fetch('/api/upload/image', {
          method: 'POST',
          body: formData
        })

        if (!response.ok) {
          throw new Error('Failed to upload image')
        }
        const data = await response.json()
        // Save the returned image ID
        setImageId(data.image_id)
        toast.success('Image uploaded successfully')
      } catch (err) {
        console.error(err)
        toast.error('Image upload failed')
        setImagePreview(null)
        setImageId(null)
      } finally {
        setIsUploading(false)
      }
    }
  }

  return (
    <div className="fixed bottom-0 left-0 w-full">
      <div className="mx-auto max-w-3xl px-2 sm:px-4 pb-4 sm:pb-6">
        <form onSubmit={handleSubmit} className="relative">
          <div className="relative rounded-2xl bg-white border border-gray-200 shadow-sm overflow-hidden">
            <Textarea
              ref={inputRef}
              name="input"
              rows={2}
              maxRows={6}
              placeholder="How can I help?"
              value={input}
              disabled={isLoading || isUploading}
              onChange={handleInputChange}
              onCompositionStart={() => setIsComposing(true)}
              onCompositionEnd={() => setIsComposing(false)}
              className="w-full resize-none bg-transparent text-gray-900 placeholder:text-gray-500 focus:outline-none py-3 sm:py-5 px-3 sm:px-5 pr-36 text-sm sm:text-base disabled:opacity-50 disabled:cursor-not-allowed"
              onKeyDown={e => {
                if (
                  e.key === 'Enter' &&
                  !e.shiftKey &&
                  !isComposing &&
                  !isLoading &&
                  !isUploading
                ) {
                  e.preventDefault()
                  const form = e.currentTarget.form
                  if (form) {
                    form.requestSubmit()
                  }
                }
              }}
            />
            <div className="absolute right-1 sm:right-2 bottom-1 sm:bottom-2 flex items-center gap-2">
              {/* Performance Analysis button */}
              <Button
                type="button"
                onClick={onShowAnalysis}
                disabled={isLoading || isUploading}
                variant="ghost"
                size="icon"
                className="bg-transparent hover:bg-gray-100 text-gray-600 hover:text-gray-900 rounded-lg p-1.5 sm:p-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <BarChart3 className="size-4 sm:size-5" />
              </Button>

              {/* Image attachment button */}
              <Button
                type="button"
                onClick={handleImageButtonClick}
                disabled={isLoading || isUploading}
                variant="ghost"
                size="icon"
                className="bg-transparent hover:bg-gray-100 text-gray-600 hover:text-gray-900 rounded-lg p-1.5 sm:p-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isUploading ? (
                  <Loader2 className="size-4 sm:size-5 animate-spin" />
                ) : (
                  <Image className="size-4 sm:size-5" />
                )}
              </Button>

              {/* Send button */}
              <Button
                type="submit"
                disabled={isLoading || isUploading || input.trim().length === 0}
                variant="ghost"
                size="icon"
                className="bg-transparent hover:bg-gray-100 text-gray-600 hover:text-gray-900 rounded-lg p-1.5 sm:p-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <Square className="size-4 sm:size-5" />
                ) : (
                  <ArrowUp className="size-4 sm:size-5" />
                )}
              </Button>
            </div>
            {/* Hidden file input */}
            <input
              type="file"
              accept="image/*"
              ref={fileInputRef}
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
          </div>
          {/* Display image preview if available */}
          {imagePreview && (
            <div className="mt-2 flex items-center gap-2">
              <img
                src={imagePreview}
                alt="Image preview"
                className="h-16 w-16 object-cover rounded"
              />
              <span className="text-sm text-gray-600">Image attached</span>
            </div>
          )}
        </form>
      </div>
    </div>
  )
}
