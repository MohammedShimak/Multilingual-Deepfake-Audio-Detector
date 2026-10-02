import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { motion, AnimatePresence } from 'framer-motion'
import { ShieldCheck, ShieldAlert, AlertTriangle, History, Calendar, Trash2, Search, Filter } from 'lucide-react'

const HistoryPage = ({ token }) => {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')

  useEffect(() => {
    fetchHistory()
  }, [])

  const fetchHistory = async () => {
    setLoading(true)
    setError(null)
    try {
      const config = {}
      if (token) {
        config.headers = { Authorization: `Bearer ${token}` }
      }
      const response = await axios.get('/api/history', config)
      setHistory(response.data)
    } catch (err) {
      console.error('Fetch history error:', err)
      setError('Could not retrieve prediction history. Check if backend is running.')
    } finally {
      setLoading(false)
    }
  }

  const filteredHistory = history.filter(item => 
    item.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
    item.result.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const formatDate = (dateString) => {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric', 
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <div className="max-w-6xl mx-auto space-y-12">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div className="space-y-2">
          <h1 className="text-5xl font-extrabold flex items-center gap-4">
            <History size={48} className="text-primary" />
            <span className="gradient-text">Analysis Log</span>
          </h1>
          <p className="text-text-muted text-lg">Tracks your recent voice authentications.</p>
        </div>
        
        <div className="flex gap-4 w-full md:w-auto">
          <div className="relative flex-grow md:w-72">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" size={18} />
            <input 
              type="text" 
              placeholder="Search filename or result..." 
              className="w-full bg-white/5 border border-white/10 rounded-xl py-2 pl-10 pr-4 text-sm focus:outline-none focus:border-primary transition-all"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <button onClick={fetchHistory} className="p-2.5 bg-white/5 border border-white/10 rounded-xl hover:bg-white/10 transition-colors">
            <Filter size={20} className="text-text-muted" />
          </button>
        </div>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[0, 1, 2, 3, 4, 5].map(i => (
            <div key={i} className="glass-card h-48 animate-pulse bg-white/5" />
          ))}
        </div>
      ) : error ? (
        <div className="glass-card p-12 text-center space-y-4">
          <ShieldAlert size={64} className="text-danger mx-auto" />
          <h3 className="text-2xl font-bold">Something went wrong</h3>
          <p className="text-text-muted">{error}</p>
          <button onClick={fetchHistory} className="primary-btn mt-4">Try Again</button>
        </div>
      ) : filteredHistory.length === 0 ? (
        <div className="glass-card p-24 text-center space-y-6">
          <div className="w-24 h-24 rounded-full bg-white/5 flex items-center justify-center mx-auto opacity-50">
            <History size={48} className="text-text-muted" />
          </div>
          <div>
            <h3 className="text-2xl font-bold">No history available</h3>
            <p className="text-text-muted">Once you analyze audio, results will appear here.</p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <AnimatePresence mode="popLayout">
            {filteredHistory.map((item, index) => {
              const isFake = item.result === 'FAKE'
              const isUncertain = item.result === 'UNCERTAIN'
              const color = isFake ? 'var(--danger)' : isUncertain ? 'var(--warning, #facc15)' : 'var(--success)'
              
              return (
                <motion.div 
                  layout
                  key={item._id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ delay: index * 0.05 }}
                  className="glass-card p-6 flex flex-col gap-6 relative group overflow-hidden"
                  style={{ borderLeft: `4px solid ${color}` }}
                >
                  <div className="flex items-start justify-between">
                    <div className="space-y-1 overflow-hidden">
                      <p className="font-bold text-lg truncate pr-2">{item.filename}</p>
                      <div className="flex items-center gap-2 text-xs text-text-muted font-medium uppercase tracking-wider">
                        <Calendar size={12} />
                        <span>{formatDate(item.created_at)}</span>
                      </div>
                    </div>
                    {isFake ? (
                      <ShieldAlert size={28} className="text-danger flex-shrink-0" />
                    ) : item.result === 'UNCERTAIN' ? (
                      <AlertTriangle size={28} className="text-warning flex-shrink-0" style={{ color: color }}/>
                    ) : (
                      <ShieldCheck size={28} className="text-success flex-shrink-0" />
                    )}
                  </div>

                  <div className="bg-white/5 rounded-xl p-4 flex items-center justify-between mt-auto">
                    <div className="space-y-1">
                      <p className="text-xs text-text-muted font-bold uppercase">Verdict</p>
                      <p className="font-black text-xl tracking-wide" style={{ color }}>{item.result}</p>
                    </div>
                    <div className="text-right space-y-1">
                      <p className="text-xs text-text-muted font-bold uppercase">Confidence</p>
                      <p className="text-2xl font-black">{Math.floor(item.confidence)}%</p>
                    </div>
                  </div>

                  <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button className="p-2 text-text-muted hover:text-danger rounded-lg hover:bg-white/5 transition-all">
                      <Trash2 size={16} />
                    </button>
                  </div>
                </motion.div>
              )
            })}
          </AnimatePresence>
        </div>
      )}
    </div>
  )
}

export default HistoryPage
