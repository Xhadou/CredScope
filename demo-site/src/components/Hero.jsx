import { motion, useMotionValue, useTransform, animate } from 'framer-motion';
import { useEffect, useState } from 'react';
import { TrendingUp, Layers, Database, Cpu, ArrowRight } from 'lucide-react';

function AnimatedCounter({ value, suffix = '', prefix = '', decimals = 0 }) {
  const count = useMotionValue(0);
  const rounded = useTransform(count, (v) => 
    prefix + v.toFixed(decimals) + suffix
  );
  const [displayValue, setDisplayValue] = useState(prefix + '0' + suffix);

  useEffect(() => {
    const controls = animate(count, value, {
      duration: 2,
      ease: 'easeOut',
    });

    const unsubscribe = rounded.on('change', (v) => setDisplayValue(v));

    return () => {
      controls.stop();
      unsubscribe();
    };
  }, [value, count, rounded, prefix, suffix]);

  return <span>{displayValue}</span>;
}

const metrics = [
  { value: 79.08, suffix: '%', label: 'AUC Score', icon: TrendingUp, color: 'from-green-400 to-emerald-500' },
  { value: 522, suffix: '', label: 'Features', icon: Layers, color: 'from-indigo-400 to-purple-500' },
  { value: 7, suffix: '', label: 'Data Sources', icon: Database, color: 'from-cyan-400 to-blue-500' },
  { value: 3, suffix: '', label: 'Ensemble Models', icon: Cpu, color: 'from-orange-400 to-pink-500' },
];

export default function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center pt-20 overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-indigo-500/20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-gradient-to-r from-indigo-500/10 to-cyan-500/10 rounded-full blur-3xl" />
      </div>
      
      {/* Grid pattern */}
      <div 
        className="absolute inset-0 opacity-20"
        style={{
          backgroundImage: `linear-gradient(rgba(99, 102, 241, 0.1) 1px, transparent 1px),
                           linear-gradient(90deg, rgba(99, 102, 241, 0.1) 1px, transparent 1px)`,
          backgroundSize: '50px 50px'
        }}
      />

      <div className="relative z-10 max-w-7xl mx-auto px-6 py-20">
        <div className="text-center">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-5xl md:text-7xl font-bold mb-6 leading-tight"
          >
            Predict Credit Risk with
            <br />
            <span className="gradient-text">Machine Learning</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-xl text-gray-400 max-w-3xl mx-auto mb-12"
          >
            CredScope uses advanced ensemble models to analyze 522 features across 7 data sources,
            delivering transparent and accurate credit risk predictions for Home Credit applicants.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="flex flex-wrap justify-center gap-4 mb-20"
          >
            <motion.a
              href="#demo"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-8 py-4 bg-gradient-to-r from-indigo-500 to-cyan-500 rounded-xl font-semibold text-lg flex items-center gap-2 shadow-lg shadow-indigo-500/25"
            >
              <span>Try Live Demo</span>
              <ArrowRight className="w-5 h-5" />
            </motion.a>
            <motion.a
              href="#architecture"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-8 py-4 glass rounded-xl font-semibold text-lg"
            >
              View Architecture
            </motion.a>
          </motion.div>

          {/* Metrics Grid */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-6"
          >
            {metrics.map((metric, index) => (
              <motion.div
                key={metric.label}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: 0.5 + index * 0.1 }}
                whileHover={{ scale: 1.05, y: -5 }}
                className="glass-card relative group"
              >
                <div className={`absolute inset-0 bg-gradient-to-br ${metric.color} opacity-0 group-hover:opacity-10 rounded-2xl transition-opacity duration-300`} />
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${metric.color} flex items-center justify-center mb-4 mx-auto`}>
                  <metric.icon className="w-6 h-6 text-white" />
                </div>
                <div className="text-4xl font-bold mb-2">
                  <AnimatedCounter 
                    value={metric.value} 
                    suffix={metric.suffix}
                    decimals={metric.suffix === '%' ? 2 : 0}
                  />
                </div>
                <div className="text-gray-400">{metric.label}</div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2"
      >
        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="w-6 h-10 rounded-full border-2 border-white/20 flex items-start justify-center p-2"
        >
          <motion.div className="w-1.5 h-1.5 rounded-full bg-white/50" />
        </motion.div>
      </motion.div>
    </section>
  );
}
