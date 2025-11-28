import { motion } from 'framer-motion';
import { useInView } from 'framer-motion';
import { useRef } from 'react';

const models = [
  { name: 'LightGBM', weight: 35.9, color: '#22c55e', description: 'Fast gradient boosting with leaf-wise growth' },
  { name: 'XGBoost', weight: 32.1, color: '#6366f1', description: 'Regularized boosting with tree pruning' },
  { name: 'CatBoost', weight: 32.0, color: '#f59e0b', description: 'Native categorical feature handling' },
];

const dataFlow = [
  { name: 'Raw Data', items: ['7 CSV Sources', '300K+ Records', '150+ Raw Features'] },
  { name: 'Feature Engineering', items: ['Aggregations', 'Time Features', 'Cross Features'] },
  { name: 'Model Training', items: ['Hyperparameter Tuning', 'Cross Validation', 'Early Stopping'] },
  { name: 'Ensemble', items: ['Weighted Average', 'Optimized Weights', '79.08% AUC'] },
];

export default function EnsembleArchitecture() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  return (
    <section id="architecture" className="py-32 relative" ref={ref}>
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center mb-20"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            Ensemble <span className="gradient-text">Architecture</span>
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Three state-of-the-art models working together, with optimized weights
            determined through cross-validation.
          </p>
        </motion.div>

        {/* Model Cards */}
        <div className="grid md:grid-cols-3 gap-6 mb-16">
          {models.map((model, index) => (
            <motion.div
              key={model.name}
              initial={{ opacity: 0, y: 40 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.5, delay: 0.2 + index * 0.1 }}
              whileHover={{ y: -10 }}
              className="glass-card relative overflow-hidden group"
            >
              {/* Glow effect */}
              <div 
                className="absolute -inset-1 opacity-0 group-hover:opacity-20 transition-opacity duration-500 blur-xl"
                style={{ background: model.color }}
              />
              
              <div className="relative z-10">
                {/* Weight indicator */}
                <div className="flex items-center justify-between mb-6">
                  <div 
                    className="w-16 h-16 rounded-2xl flex items-center justify-center text-2xl font-bold"
                    style={{ background: `${model.color}20`, color: model.color }}
                  >
                    {model.weight}%
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-gray-500">Weight</div>
                    <div className="text-lg font-semibold">{model.weight}%</div>
                  </div>
                </div>

                <h3 className="text-2xl font-bold mb-2">{model.name}</h3>
                <p className="text-gray-400">{model.description}</p>

                {/* Progress bar */}
                <div className="mt-6">
                  <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={isInView ? { width: `${model.weight}%` } : {}}
                      transition={{ duration: 1, delay: 0.5 + index * 0.2 }}
                      className="h-full rounded-full"
                      style={{ background: model.color }}
                    />
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Data Flow Pipeline */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="glass-card"
        >
          <h3 className="text-2xl font-bold mb-8 text-center">Data Pipeline</h3>
          <div className="relative">
            {/* Connection lines */}
            <div className="hidden md:block absolute top-5 left-[12.5%] right-[12.5%] h-0.5 bg-gradient-to-r from-indigo-500/50 via-cyan-500/50 to-indigo-500/50" />
            
            <div className="grid md:grid-cols-4 gap-6">
              {dataFlow.map((stage, index) => (
                <motion.div
                  key={stage.name}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={isInView ? { opacity: 1, scale: 1 } : {}}
                  transition={{ duration: 0.4, delay: 0.6 + index * 0.15 }}
                  className="relative"
                >
                  {/* Step number */}
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-cyan-500 flex items-center justify-center text-lg font-bold mb-4 mx-auto relative z-10">
                    {index + 1}
                  </div>
                  
                  <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                    <h4 className="font-semibold text-center mb-3">{stage.name}</h4>
                    <ul className="space-y-2">
                      {stage.items.map((item) => (
                        <li key={item} className="text-sm text-gray-400 text-center">
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Final Score */}
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={isInView ? { opacity: 1, scale: 1 } : {}}
          transition={{ duration: 0.5, delay: 1 }}
          className="mt-12 text-center"
        >
          <div className="inline-flex items-center gap-4 px-8 py-4 glass rounded-2xl">
            <span className="text-gray-400">Final Ensemble AUC:</span>
            <span className="text-4xl font-bold gradient-text">79.08%</span>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
