import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileAudio, CheckCircle, XCircle, Loader2, AlertCircle } from 'lucide-react'

const AudioUpload = ({ onUpload, loading }) => {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [error, setError] = useState(null)

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    setError(null)
    
    if (rejectedFiles?.length > 0) {
      const reason = rejectedFiles[0].errors[0].code
      if (reason === 'file-invalid-type') {
        setError("Unsupported format. Please upload WAV, MP3, OGG, or FLAC.")
      } else if (reason === 'file-too-large') {
        setError("File is too large. Please upload audio under 5MB.")
      } else {
        setError("Invalid file selection.")
      }
      return
    }

    if (acceptedFiles?.length > 0) {
      const selectedFile = acceptedFiles[0]
      setFile(selectedFile)
      if (preview) URL.revokeObjectURL(preview)
      setPreview(URL.createObjectURL(selectedFile))
    }
  }, [preview])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: { 
      'audio/wav': ['.wav'],
      'audio/mpeg': ['.mp3'],
      'audio/ogg': ['.ogg'],
      'audio/flac': ['.flac'],
      'audio/x-m4a': ['.m4a']
    },
    multiple: false,
    maxSize: 5242880, // 5MB
    disabled: loading
  })

  const handleStartAnalysis = () => {
    if (file && onUpload) {
      onUpload(file)
    }
  }

  const cancelSelection = () => {
    setFile(null)
    if (preview) URL.revokeObjectURL(preview)
    setPreview(null)
    setError(null)
  }

  return (
    <div className="flex flex-col gap-8 w-full">
      <div 
        {...getRootProps()} 
        className={`relative border-2 border-dashed rounded-3xl p-10 transition-all flex flex-col items-center gap-6 cursor-pointer group 
        ${isDragActive ? 'border-primary bg-primary/5 scale-[1.01]' : 'border-white/10 hover:border-white/20 hover:bg-white/5'}
        ${isDragReject ? 'border-danger bg-danger/5' : ''}
        ${loading ? 'opacity-50 pointer-events-none' : ''}`}
      >
        <input {...getInputProps()} />
        <div className="w-16 h-16 bg-primary/10 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform">
          {loading ? (
            <Loader2 className="animate-spin text-primary" size={32} />
          ) : file ? (
            <CheckCircle className="text-success" size={32} />
          ) : (
            <Upload className="text-primary" size={32} />
          )}
        </div>
        
        <div className="text-center space-y-1">
          <p className="text-xl font-bold">
            {isDragActive ? 'Drop here' : file ? 'Audio Selected' : 'Select voice sample'}
          </p>
          <p className="text-sm text-text-muted">
            {file ? file.name : 'Supports WAV, MP3, OGG, & FLAC (<5MB)'}
          </p>
        </div>
        
        {!loading && (
          <button type="button" className="px-5 py-2 bg-white/10 hover:bg-white/20 transition-colors rounded-full text-xs font-semibold">
            {file ? 'Change File' : 'Browse Computer'}
          </button>
        )}
      </div>

      {error && (
        <div className="bg-danger/10 border border-danger/20 p-4 rounded-2xl flex items-center gap-3 text-danger text-sm animate-in fade-in zoom-in duration-300">
          <AlertCircle size={18} />
          {error}
        </div>
      )}

      {file && !loading && (
        <div className="bg-white/5 rounded-2xl p-6 border border-white/10 flex flex-col gap-6 animate-in slide-in-from-bottom duration-300">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-primary/10 rounded-xl text-primary">
                <FileAudio size={20} />
              </div>
              <div className="overflow-hidden">
                <p className="font-semibold truncate max-w-[150px]">{file.name}</p>
                <p className="text-xs text-text-muted">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
              </div>
            </div>
            <button onClick={cancelSelection} className="p-2 hover:bg-white/10 rounded-full text-text-muted transition-all">
              <XCircle size={20} />
            </button>
          </div>
          
          {preview && (
            <div className="rounded-xl overflow-hidden bg-black/40 p-2">
              <audio src={preview} controls className="w-full h-8" />
            </div>
          )}
          
          <button 
            onClick={handleStartAnalysis} 
            className="primary-btn w-full text-md flex items-center justify-center gap-2 py-3"
          >
            Launch Deepfake Analysis
            <CheckCircle size={18} />
          </button>
        </div>
      )}
      
      {loading && (
        <div className="p-10 flex flex-col items-center justify-center gap-6 bg-white/5 rounded-2xl border border-primary/20">
          <div className="flex items-center gap-1 h-8">
            {[...Array(12)].map((_, i) => (
              <div 
                key={i} 
                className="w-1.5 bg-primary rounded-full animate-bounce" 
                style={{ 
                  height: '100%',
                  animationDelay: `${i * 0.1}s`,
                  animationDuration: '0.8s'
                }} 
              />
            ))}
          </div>
          <div className="text-center">
            <h3 className="text-lg font-bold text-primary animate-pulse">Running Neural Classifiers...</h3>
            <p className="text-xs text-text-muted uppercase tracking-tighter">Analyzing spectral micro-fluctuations</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default AudioUpload
