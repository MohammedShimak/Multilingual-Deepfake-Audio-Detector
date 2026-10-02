import React from 'react'
import { motion } from 'framer-motion'
import { ShieldCheck, ShieldAlert, AlertTriangle, Cpu, Activity, InfoIcon, BarChart3 } from 'lucide-react'

const ResultCard = ({ data }) => {
  if (!data) return null;

  const isFake = data.result === 'FAKE'
  const isUncertain = data.result === 'UNCERTAIN'
  const isHuman = data.result === 'REAL'
  
  // Dynamic styling based on stabilized types
  let accentColor = '#10b981' // Green
  let icon = <ShieldCheck size={64} />
  
  if (isFake) {
    accentColor = '#ef4444' // Red
    icon = <ShieldAlert size={64} />
  } else if (isUncertain) {
    accentColor = '#facc15' // Yellow
    icon = <AlertTriangle size={64} />
  }

  const bgGradient = `linear-gradient(135deg, ${accentColor}15 0%, ${accentColor}05 100%)`

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="glass-card overflow-hidden flex flex-col p-8 relative border-2"
      style={{ borderColor: `${accentColor}40`, background: bgGradient }}
    >
      <div className="flex flex-col md:flex-row gap-8 items-center">
        <div className="relative">
          <motion.div 
            animate={isUncertain ? { scale: [1, 1.05, 1] } : {}}
            transition={{ repeat: Infinity, duration: 2 }}
            className="w-32 h-32 rounded-3xl flex items-center justify-center bg-white/5 border border-white/10 shadow-2xl"
            style={{ color: accentColor }}
          >
            {icon}
          </motion.div>
          <div className="absolute -bottom-2 -right-2 bg-black border border-white/20 p-2 rounded-xl">
             <Activity size={16} style={{ color: accentColor }} />
          </div>
        </div>

        <div className="flex-1 space-y-4 text-center md:text-left">
          <div>
            <div className="flex items-center justify-center md:justify-start gap-2 opacity-50 mb-1">
                <BarChart3 size={12} />
                <span className="text-[10px] uppercase font-black tracking-widest">Neural Stability Filter Active</span>
            </div>
            <h2 className="text-4xl font-black" style={{ color: accentColor }}>
              {data.result}
            </h2>
          </div>
          
          <div className="grid grid-cols-1 gap-4">
            <div className="bg-black/20 p-4 rounded-2xl border border-white/5">
              <p className="text-[10px] uppercase font-bold opacity-40 mb-1">Confidence Grade</p>
              <p className="text-xl font-black">
                {data.confidence_level} 
                <span className="text-xs ml-1 opacity-60">({Math.round(data.confidence)}%)</span>
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-8 pt-6 border-t border-white/10 space-y-4">
        <div className="flex gap-4 bg-white/5 p-4 rounded-2xl border border-white/5 items-start">
          <div className="mt-1 flex-shrink-0">
            <InfoIcon size={20} style={{ color: accentColor }} />
          </div>
          <div>
            <p className="text-xs font-black uppercase opacity-60 mb-1">Classifier Insight</p>
            <p className="text-sm leading-relaxed font-medium">
              {data.explanation}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 px-2">
          <Cpu size={14} className="opacity-30" />
          <p className="text-[9px] uppercase tracking-[0.2em] font-bold opacity-30">
            CNN + Gradient Boosting Authentication Model
          </p>
        </div>
      </div>
    </motion.div>
  )
}

export default ResultCard
