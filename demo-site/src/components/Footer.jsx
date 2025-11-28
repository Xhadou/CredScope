import { motion } from 'framer-motion';
import { Shield, Github, Linkedin, Mail, ExternalLink } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="py-16 border-t border-white/10">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid md:grid-cols-4 gap-12 mb-12">
          {/* Brand */}
          <div className="md:col-span-2">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-cyan-400 flex items-center justify-center">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold">CredScope</span>
            </div>
            <p className="text-gray-400 max-w-md">
              Enterprise-grade credit risk assessment powered by machine learning. 
              Built for transparency, accuracy, and regulatory compliance.
            </p>
          </div>

          {/* Links */}
          <div>
            <h4 className="font-semibold mb-4">Resources</h4>
            <ul className="space-y-2 text-gray-400">
              <li><a href="#features" className="hover:text-white transition-colors">Features</a></li>
              <li><a href="#performance" className="hover:text-white transition-colors">Performance</a></li>
              <li><a href="#demo" className="hover:text-white transition-colors">Live Demo</a></li>
              <li><a href="#architecture" className="hover:text-white transition-colors">Architecture</a></li>
            </ul>
          </div>

          {/* Tech */}
          <div>
            <h4 className="font-semibold mb-4">Technology</h4>
            <ul className="space-y-2 text-gray-400">
              <li>LightGBM</li>
              <li>XGBoost</li>
              <li>CatBoost</li>
              <li>SHAP Explainability</li>
            </ul>
          </div>
        </div>

        {/* Bottom */}
        <div className="pt-8 border-t border-white/10 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-gray-500 text-sm">
            © 2025 CredScope. Built for Home Credit Default Risk Challenge.
          </p>
          
          <div className="flex items-center gap-4">
            <motion.a
              href="https://github.com/Xhadou/CredScope-v1"
              target="_blank"
              rel="noopener noreferrer"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              className="p-2 rounded-lg hover:bg-white/10 transition-colors text-gray-400 hover:text-white"
            >
              <Github className="w-5 h-5" />
            </motion.a>
            <motion.a
              href="https://www.linkedin.com/in/pratyuush/"
              target="_blank"
              rel="noopener noreferrer"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              className="p-2 rounded-lg hover:bg-white/10 transition-colors text-gray-400 hover:text-white"
            >
              <Linkedin className="w-5 h-5" />
            </motion.a>
            <motion.a
              href="mailto:pj825@snu.edu.in"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              className="p-2 rounded-lg hover:bg-white/10 transition-colors text-gray-400 hover:text-white"
            >
              <Mail className="w-5 h-5" />
            </motion.a>
          </div>
        </div>
      </div>
    </footer>
  );
}
