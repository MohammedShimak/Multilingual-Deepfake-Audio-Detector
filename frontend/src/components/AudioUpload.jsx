import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, Music, FileAudio, CheckCircle, XCircle, Loader2 } from 'lucide-react'

const AudioUpload = ({ onUpload, loading }) => {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles?.length > 0) {
      const selectedFile = acceptedFiles[0]
      setFile(selectedFile)
      // Revoke old object URL if any
      if (preview) URL.revokeObjectURL(preview)
      setPreview(URL.createObjectURL(selectedFile))
    }
  }, [preview])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: { 'audio/*': ['.wav', '.mp3', '.flac', '.ogg'] },
    multiple: false,
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
  }

  return (
    <div className="flex flex-col gap-8 w-full">
      <div 
        {...getRootProps()} 
        className={`relative border-2 border-dashed rounded-3xl p-12 transition-all flex flex-col items-center gap-6 cursor-pointer group 
        ${isDragActive ? 'border-primary bg-primary/5 scale-[1.01]' : 'border-white/10 hover:border-white/20 hover:bg-white/5'}
        ${isDragReject ? 'border-danger bg-danger/5' : ''}
        ${loading ? 'opacity-50 pointer-events-none' : ''}`}
      >
        <input {...getInputProps()} />
        <div className="w-20 h-20 bg-primary/10 rounded-2xl flex items-center justify-center group-hover:scale-110 group-hover:rotate-6 transition-transform">
          {loading ? (
            <Loader2 className="animate-spin text-primary" size={40} />
          ) : file ? (
            <CheckCircle className="text-success animate-in zoom-in duration-300" size={40} />
          ) : (
            <Upload className="text-primary" size={40} />
          )}
        </div>
        
        <div className="text-center space-y-2">
          <p className="text-2xl font-bold flex items-center justify-center gap-2">
            {isDragActive 
              ? 'Drop voice sample here' 
              : file 
                ? 'Audio Selected! ✅' 
                : 'Select audio for verification'}
          </p>
          <p className="text-text-muted">
            {file ? 'Click the button below to analyze' : 'WAV, MP3, FLAC or OGG (max 20MB)'}
          </p>
        </div>
        
        <button className="px-6 py-2 bg-white/10 rounded-full text-sm font-semibold hover:bg-white/20 transition-colors">
          {file ? 'Change File' : 'Browse Files'}
        </button>      </div>

      {file && !loading && (
        <div className="bg-white/5 rounded-2xl p-6 border border-white/10 flex flex-col gap-6 animate-in fade-in slide-in-from-bottom duration-300">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-secondary/10 rounded-xl text-secondary">
                <FileAudio size={24} />
              </div>
              <div className="overflow-hidden">
                <p className="font-semibold truncate max-w-[200px] sm:max-w-md">{file.name}</p>
                <p className="text-xs text-text-muted">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
              </div>
            </div>
            <button onClick={cancelSelection} className="p-2 hover:bg-white/10 rounded-full text-text-muted hover:text-danger flex-shrink-0 transition-all">
              <XCircle size={24} />
            </button>
          </div>
          
          {preview && (
            <div className="rounded-xl overflow-hidden bg-black/20 p-4">
              <audio src={preview} controls className="w-full h-10" />
            </div>
          )}
          
          <button 
            onClick={handleStartAnalysis} 
            className="primary-btn w-full text-lg flex items-center justify-center gap-3 py-4"
          >
            Start AI Verification
            <CheckCircle size={20} />
          </button>
        </div>
      )}
      
      {loading && (
        <div className="p-10 flex flex-col items-center justify-center gap-8 bg-white/5 rounded-2xl border border-primary/20 animate-in fade-in zoom-in duration-500">
          <div className="relative w-full max-w-[200px] h-16 flex items-end justify-between gap-1">
            {[...Array(20)].map((_, i) => (
              <div 
                key={i} 
                className="w-full bg-gradient-to-t from-primary to-secondary rounded-full" 
                style={{ 
                  height: '20%',
                  animation: `waveform 1.2s ease-in-out infinite alternate ${i * 0.05}s`
                }} 
              />
            ))}
          </div>
          <style dangerouslySetInnerHTML={{__html: `
            @keyframes waveform {
              0% { height: 10%; opacity: 0.5; }
              100% { height: 100%; opacity: 1; }
            }
          `}} />
          
          <div className="text-center space-y-2">
            <h3 className="text-2xl font-black bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent animate-pulse">
              Analyzing Voice Patterns...
            </h3>
            <p className="text-text-muted text-sm">Extracting Mel-spectrograms & running AI classifiers</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default AudioUpload
