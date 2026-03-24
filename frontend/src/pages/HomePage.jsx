import React, { useState } from 'react'
import { Upload, AudioWaveform, RotateCcw, ShieldCheck, ShieldAlert } from 'lucide-react'
import axios from 'axios'
import AudioUpload from '../components/AudioUpload'
import ResultCard from '../components/ResultCard'

const HomePage = ({ token }) => {
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleUpload = async (file) => {
    setLoading(true)
    setError(null)
    setPrediction(null)
    
    const formData = new FormData()
    formData.append('file', file)

    try {
      const config = { headers: { 'Content-Type': 'multipart/form-data' } }
      if (token) {
        config.headers.Authorization = `Bearer ${token}`
      }
      
      const response = await axios.post('http://localhost:8000/api/predict', formData, config)
      setPrediction(response.data)
    } catch (err) {
      console.error('Prediction error:', err)
      setError(err.response?.data?.detail || 'Failed to analyze audio. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setPrediction(null)
    setError(null)
  }

  return (
    <div className="flex flex-col items-center gap-12 max-w-4xl mx-auto w-full">
      <div className="text-center space-y-4 animate-in fade-in slide-in-from-top duration-700">
        <h1 className="text-5xl md:text-6xl lg:text-7xl font-extrabold tracking-tight">
          Multilingual <span className="gradient-text">Deepfake Audio Detector</span>
        </h1>
        <p className="text-text-muted text-lg max-w-xl mx-auto">
          Our advanced AI analyzes spectral features and deep neural patterns to distinguish 
          real voices from synthetic deepfakes in seconds.
        </p>
      </div>

      {!prediction ? (
        <div className="w-full glass-card p-8 md:p-12 animate-in fade-in zoom-in duration-500 delay-200">
          <AudioUpload onUpload={handleUpload} loading={loading} />
          {error && (
            <div className="mt-6 p-4 rounded-xl bg-danger/10 border border-danger/20 text-danger text-center flex items-center justify-center gap-2">
              <ShieldAlert size={20} />
              <span>{error}</span>
            </div>
          )}
        </div>
      ) : (
        <div className="w-full grid md:grid-cols-2 gap-8 animate-in fade-in slide-in-from-bottom duration-700">
          <ResultCard data={prediction} />
          <div className="glass-card p-8 flex flex-col justify-center items-center text-center space-y-6">
            <div className="p-4 bg-white/5 rounded-full">
              <RotateCcw size={40} className="text-primary hover:rotate-[-45deg] transition-transform cursor-pointer" onClick={reset} />
            </div>
            <h3 className="text-2xl font-bold">New Analysis?</h3>
            <p className="text-text-muted">Analyze another audio sample to continue verification.</p>
            <button onClick={reset} className="primary-btn w-full">
              Restart
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default HomePage
