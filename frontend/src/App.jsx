import React, { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import { Home, History, Shield, Github, User } from 'lucide-react'
import HomePage from './pages/HomePage'
import HistoryPage from './pages/HistoryPage'
import AuthModal from './components/AuthModal'

const Navbar = ({ user, handleLogout, setIsAuthOpen }) => {
  const location = useLocation()
  
  return (
    <nav className="glass-card mt-6 mx-4 px-6 py-4 flex items-center justify-between sticky top-6 z-50">
      <div className="flex items-center gap-3">
        <div className="bg-primary p-2 rounded-xl">
          <Shield color="white" size={24} />
        </div>
        <span className="text-xl font-bold gradient-text">AudioVerifier</span>
      </div>
      
      <div className="flex bg-white/5 rounded-2xl p-1 gap-2 border border-white/10">
        <Link to="/" className={`px-5 py-2 rounded-xl text-sm font-medium transition-all ${location.pathname === '/' ? 'bg-primary text-white shadow-lg' : 'hover:bg-white/5 text-text-muted'}`}>
          <div className="flex items-center gap-2">
            <Home size={18} />
            <span>Detector</span>
          </div>
        </Link>
        <Link to="/history" className={`px-5 py-2 rounded-xl text-sm font-medium transition-all ${location.pathname === '/history' ? 'bg-primary text-white shadow-lg' : 'hover:bg-white/5 text-text-muted'}`}>
          <div className="flex items-center gap-2">
            <History size={18} />
            <span>History</span>
          </div>
        </Link>
      </div>

      <div className="hidden sm:flex items-center gap-4">
        {user ? (
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 bg-primary/10 px-3 py-1.5 rounded-full border border-primary/20">
              <User size={16} className="text-primary" />
              <span className="text-sm font-medium text-white">{user.username}</span>
            </div>
            <button onClick={handleLogout} className="text-sm font-semibold text-text-muted hover:text-danger transition-colors">
              Logout
            </button>
          </div>
        ) : (
          <button 
            onClick={() => setIsAuthOpen(true)} 
            className="flex items-center gap-2 bg-white/10 hover:bg-white/20 px-4 py-2 rounded-xl text-white font-semibold transition-colors"
          >
            <User size={18} />
            <span>Login</span>
          </button>
        )}
      </div>
    </nav>
  )
}

function App() {
  const [token, setToken] = useState(localStorage.getItem('token') || null);
  const [user, setUser] = useState(JSON.parse(localStorage.getItem('user')) || null);
  const [isAuthOpen, setIsAuthOpen] = useState(false);

  const handleLogin = (userData, jwtToken) => {
    setUser(userData);
    setToken(jwtToken);
    localStorage.setItem('user', JSON.stringify(userData));
    localStorage.setItem('token', jwtToken);
  };

  const handleLogout = () => {
    setUser(null);
    setToken(null);
    localStorage.removeItem('user');
    localStorage.removeItem('token');
  };

  return (
    <Router>
      <div className="flex flex-col min-h-screen">
        <Navbar user={user} handleLogout={handleLogout} setIsAuthOpen={setIsAuthOpen} />
        <main className="flex-grow container mx-auto px-4 py-12">
          <Routes>
            <Route path="/" element={<HomePage token={token} />} />
            <Route path="/history" element={<HistoryPage token={token} />} />
          </Routes>
        </main>
        
        <footer className="py-12 bg-black/20 border-t border-white/5 mt-auto">
          <div className="container mx-auto px-4 text-center flex flex-col items-center space-y-4">
            <h4 className="text-xl font-bold gradient-text">About AudioVerifier</h4>
            <p className="text-text-muted text-sm max-w-2xl">
              AudioVerifier leverages a cutting-edge CNN + Gradient Boosting deep learning model with 98% accuracy 
              to analyze Mel-spectrograms of your audio. The CNN layers extract spatial features from spectrograms while 
              the XGBoost captures temporal patterns, enabling precise detection of synthetic frequencies 
              and AI-generated artifacts to distinguish authentic human voices from deepfakes in seconds.
            </p>
            <div className="pt-6 mt-2 border-t border-white/5 w-64">
              <p className="text-text-muted text-xs">
                Deepfake Detector
              </p>
            </div>
          </div>
        </footer>
        <AuthModal isOpen={isAuthOpen} onClose={() => setIsAuthOpen(false)} onLogin={handleLogin} />
      </div>
    </Router>
  )
}

export default App
