import { motion } from 'framer-motion';
import { Shield, Github, ExternalLink, FlaskConical } from 'lucide-react';

export default function Navbar({ onNavigate, currentPage }) {
  const handleNavClick = (e, section) => {
    if (currentPage !== 'home') {
      e.preventDefault();
      onNavigate('home');
      // Scroll to section after navigation
      setTimeout(() => {
        const element = document.querySelector(section);
        if (element) element.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    }
  };

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      className="fixed top-0 left-0 right-0 z-50 glass border-b border-white/10"
    >
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <motion.button 
            onClick={() => onNavigate('home')}
            className="flex items-center gap-3"
            whileHover={{ scale: 1.02 }}
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-cyan-400 flex items-center justify-center">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold">CredScope</span>
          </motion.button>
          
          <div className="hidden md:flex items-center gap-8">
            <a href="#features" onClick={(e) => handleNavClick(e, '#features')} className="text-gray-400 hover:text-white transition-colors">Features</a>
            <a href="#performance" onClick={(e) => handleNavClick(e, '#performance')} className="text-gray-400 hover:text-white transition-colors">Performance</a>
            <a href="#demo" onClick={(e) => handleNavClick(e, '#demo')} className="text-gray-400 hover:text-white transition-colors">Demo</a>
            <a href="#architecture" onClick={(e) => handleNavClick(e, '#architecture')} className="text-gray-400 hover:text-white transition-colors">Architecture</a>
          </div>
          
          <div className="flex items-center gap-4">
            <motion.a
              href="https://github.com/Xhadou/CredScope-v1"
              target="_blank"
              rel="noopener noreferrer"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="p-2 rounded-lg hover:bg-white/10 transition-colors"
            >
              <Github className="w-5 h-5" />
            </motion.a>
            <motion.button
              onClick={() => onNavigate('predict')}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-4 py-2 bg-gradient-to-r from-indigo-500 to-cyan-500 rounded-lg font-medium flex items-center gap-2"
            >
              <span>Try API</span>
              <ExternalLink className="w-4 h-4" />
            </motion.button>
          </div>
        </div>
      </div>
    </motion.nav>
  );
}
