import { useState } from 'react';
import axios from 'axios';
import { X, Loader2 } from 'lucide-react';

export default function AuthModal({ isOpen, onClose, onLogin }) {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const endpoint = isLogin ? '/api/auth/login' : '/api/auth/register';
      const res = await axios.post(`http://localhost:8000${endpoint}`, { username, password });
      onLogin(res.data.user, res.data.token);
      onClose();
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred. Make sure backend is running & MongoDB is active.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200">
      <div className="bg-[#12141c] w-full max-w-md rounded-2xl border border-white/10 p-8 relative shadow-2xl">
        <button onClick={onClose} className="absolute top-4 right-4 text-text-muted hover:text-white transition-colors">
          <X size={24} />
        </button>

        <h2 className="text-3xl font-bold mb-6 gradient-text">{isLogin ? 'Welcome Back' : 'Create Account'}</h2>

        {error && <div className="bg-danger/20 text-danger text-sm p-3 rounded-lg mb-4">{error}</div>}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-text-muted mb-1">Username</label>
            <input 
              type="text" 
              required
              value={username} 
              onChange={e => setUsername(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors" 
              placeholder="" 
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-text-muted mb-1">Password</label>
            <input 
              type="password" 
              required
              value={password} 
              onChange={e => setPassword(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors" 
              placeholder="••••••••" 
            />
          </div>

          <button 
            type="submit" 
            disabled={loading}
            className="w-full bg-primary hover:bg-primary-hover text-white font-bold py-3 rounded-lg mt-6 flex justify-center items-center transition-colors disabled:opacity-50"
          >
            {loading ? <Loader2 className="animate-spin" /> : isLogin ? 'Login' : 'Sign Up'}
          </button>
        </form>

        <p className="text-center text-text-muted text-sm mt-6">
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <button 
            onClick={() => { setIsLogin(!isLogin); setError(''); }} 
            className="text-primary hover:text-primary-hover font-semibold transition-colors"
          >
            {isLogin ? 'Sign up' : 'Login'}
          </button>
        </p>
      </div>
    </div>
  );
}
