import React from 'react'
import { motion } from 'framer-motion'
import { ShieldCheck, ShieldAlert, FileAudio, Info, Calendar } from 'lucide-react'

const ResultCard = ({ data }) => {
  const isFake = data.result === 'FAKE'
  const accentColor = isFake ? 'var(--danger)' : 'var(--success)'
  const bgGradient = isFake 
    ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.02) 100%)'
    : 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.02) 100%)'
  
  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.9, y: 30 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.6, type: 'spring', stiffness: 100 }}
      className="glass-card overflow-hidden flex flex-col items-center justify-center p-10 relative group"
      style={{ border: `1px solid ${accentColor}20`, background: bgGradient }}
    >
      <div className="absolute top-6 left-6 flex items-center gap-2 px-3 py-1 bg-white/5 rounded-full border border-white/10">
        <FileAudio size={16} className="text-text-muted" />
        <span className="text-xs text-text-muted truncate max-w-[120px]">{data.filename}</span>
      </div>

      <div className="flex flex-col items-center text-center gap-8 py-6 w-full">
        <motion.div 
          initial={{ rotate: -15, scale: 0.8 }}
          animate={{ rotate: 0, scale: 1 }}
          transition={{ delay: 0.3, type: 'spring' }}
          className="w-40 h-40 bg-white/5 rounded-full flex items-center justify-center relative"
        >
          {/* Animated rings */}
          <motion.div 
            animate={{ scale: [1, 1.1, 1], opacity: [0.3, 0.1, 0.3] }}
            transition={{ duration: 3, repeat: Infinity }}
            className="absolute inset-0 rounded-full border-4 border-dashed"
            style={{ borderColor: `${accentColor}40` }}
          />
          <motion.div 
            animate={{ scale: [1, 1.25, 1], opacity: [0.2, 0.05, 0.2] }}
            transition={{ duration: 5, repeat: Infinity }}
            className="absolute -inset-4 rounded-full border-2 border-dashed"
            style={{ borderColor: `${accentColor}20` }}
          />
          
          {isFake ? (
            <ShieldAlert size={80} style={{ color: accentColor }} />
          ) : (
            <ShieldCheck size={80} style={{ color: accentColor }} />
          )}
        </motion.div>

        <div className="space-y-4">
          <motion.h2 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="text-4xl font-extrabold"
            style={{ color: accentColor }}
          >
            {isFake ? 'Deepfake Detected' : 'Verified Human'}
          </motion.h2>
          
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-1 items-center">
              <span className="text-sm font-medium text-text-muted uppercase tracking-widest">Confidence Score</span>
              <div className="flex items-center gap-3">
                <span className="text-6xl font-black">{Math.floor(data.confidence)}%</span>
                <div className="w-1.5 h-12 bg-white/10 rounded-full overflow-hidden flex flex-col-reverse">
                  <motion.div 
                    initial={{ height: 0 }}
                    animate={{ height: `${data.confidence}%` }}
                    transition={{ duration: 1, delay: 0.8 }}
                    className="w-full"
                    style={{ backgroundColor: accentColor }}
                  />
                </div>
              </div>
            </div>
            
            <motion.p 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1 }}
              className="text-sm text-text-muted max-w-[280px]"
            >
              {isFake 
                ? "Our system found high similarity with AI-generated patterns from models like VALLE or XTTS." 
                : "The voice exhibits natural human micro-fluctuations and spectral patterns typical of biological vocal cords."}
            </motion.p>
          </div>
        </div>
      </div>

      <div className="absolute bottom-6 right-6 p-2 rounded-xl hover:bg-white/10 transition-colors cursor-help">
        <Info size={18} className="text-text-muted" />
      </div>
    </motion.div>
  )
}

export default ResultCard
