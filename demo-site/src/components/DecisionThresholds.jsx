import { motion } from 'framer-motion';
import { useInView } from 'framer-motion';
import { useRef } from 'react';
import { CheckCircle, AlertTriangle, XCircle, ArrowRight } from 'lucide-react';

const thresholds = [
  {
    name: 'APPROVE',
    range: '0% - 20%',
    icon: CheckCircle,
    color: '#22c55e',
    bg: 'from-green-500/20 to-green-500/5',
    description: 'Low risk applications are automatically approved. These applicants have strong credit histories and high external scores.',
    action: 'Automatic Approval',
    examples: ['Strong external credit scores (>0.6)', 'Stable employment history', 'Reasonable debt-to-income ratio']
  },
  {
    name: 'REVIEW',
    range: '20% - 50%',
    icon: AlertTriangle,
    color: '#f59e0b',
    bg: 'from-amber-500/20 to-amber-500/5',
    description: 'Medium risk applications require human review. Credit officers examine additional factors before making a decision.',
    action: 'Manual Review Required',
    examples: ['Mixed credit signals', 'Recent employment changes', 'Borderline debt ratios']
  },
  {
    name: 'REJECT',
    range: '50% - 100%',
    icon: XCircle,
    color: '#ef4444',
    bg: 'from-red-500/20 to-red-500/5',
    description: 'High risk applications are declined. The model identifies significant default risk factors.',
    action: 'Automatic Rejection',
    examples: ['Low external credit scores (<0.3)', 'High debt-to-income ratio', 'Limited credit history']
  },
];

export default function DecisionThresholds() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  return (
    <section className="py-32 relative" ref={ref}>
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center mb-20"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            Decision <span className="gradient-text">Thresholds</span>
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Transparent risk categories ensure consistent decision-making and regulatory compliance.
          </p>
        </motion.div>

        {/* Threshold Visualization */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mb-16"
        >
          <div className="glass-card">
            <h3 className="text-xl font-semibold mb-6 text-center">Risk Score Spectrum</h3>
            
            {/* Visual bar */}
            <div className="relative h-16 rounded-2xl overflow-hidden mb-4">
              <div className="absolute inset-0 flex">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={isInView ? { width: '20%' } : {}}
                  transition={{ duration: 0.8, delay: 0.3 }}
                  className="bg-gradient-to-r from-green-600 to-green-500 flex items-center justify-center"
                >
                  <span className="text-sm font-semibold">APPROVE</span>
                </motion.div>
                <motion.div 
                  initial={{ width: 0 }}
                  animate={isInView ? { width: '30%' } : {}}
                  transition={{ duration: 0.8, delay: 0.5 }}
                  className="bg-gradient-to-r from-amber-500 to-amber-400 flex items-center justify-center"
                >
                  <span className="text-sm font-semibold text-gray-900">REVIEW</span>
                </motion.div>
                <motion.div 
                  initial={{ width: 0 }}
                  animate={isInView ? { width: '50%' } : {}}
                  transition={{ duration: 0.8, delay: 0.7 }}
                  className="bg-gradient-to-r from-red-500 to-red-600 flex items-center justify-center"
                >
                  <span className="text-sm font-semibold">REJECT</span>
                </motion.div>
              </div>
            </div>
            
            {/* Labels */}
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">0%</span>
              <span className="text-green-400 font-medium">20%</span>
              <span className="text-amber-400 font-medium">50%</span>
              <span className="text-gray-400">100%</span>
            </div>
          </div>
        </motion.div>

        {/* Threshold Cards */}
        <div className="grid md:grid-cols-3 gap-6">
          {thresholds.map((threshold, index) => (
            <motion.div
              key={threshold.name}
              initial={{ opacity: 0, y: 40 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.5, delay: 0.3 + index * 0.15 }}
              whileHover={{ y: -10 }}
              className="glass-card relative overflow-hidden group"
            >
              {/* Gradient background */}
              <div className={`absolute inset-0 bg-gradient-to-b ${threshold.bg} opacity-50`} />
              
              <div className="relative z-10">
                {/* Header */}
                <div className="flex items-center justify-between mb-6">
                  <div 
                    className="w-14 h-14 rounded-xl flex items-center justify-center"
                    style={{ background: `${threshold.color}20` }}
                  >
                    <threshold.icon className="w-7 h-7" style={{ color: threshold.color }} />
                  </div>
                  <div 
                    className="px-3 py-1.5 rounded-lg text-sm font-bold"
                    style={{ background: `${threshold.color}20`, color: threshold.color }}
                  >
                    {threshold.range}
                  </div>
                </div>

                {/* Title */}
                <h3 className="text-2xl font-bold mb-2" style={{ color: threshold.color }}>
                  {threshold.name}
                </h3>
                
                {/* Action */}
                <div className="flex items-center gap-2 mb-4">
                  <ArrowRight className="w-4 h-4 text-gray-500" />
                  <span className="text-gray-300 font-medium">{threshold.action}</span>
                </div>

                {/* Description */}
                <p className="text-gray-400 mb-6">{threshold.description}</p>

                {/* Examples */}
                <div className="space-y-2">
                  <h4 className="text-sm font-medium text-gray-300">Common Indicators:</h4>
                  {threshold.examples.map((example, i) => (
                    <div key={i} className="flex items-center gap-2 text-sm text-gray-500">
                      <div 
                        className="w-1.5 h-1.5 rounded-full"
                        style={{ background: threshold.color }}
                      />
                      <span>{example}</span>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.8 }}
          className="mt-12 glass-card"
        >
          <div className="grid md:grid-cols-3 gap-6 text-center">
            {[
              { label: 'Auto-Approved', value: '~45%', desc: 'of applications', color: '#22c55e' },
              { label: 'Manual Review', value: '~35%', desc: 'of applications', color: '#f59e0b' },
              { label: 'Auto-Rejected', value: '~20%', desc: 'of applications', color: '#ef4444' },
            ].map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={isInView ? { opacity: 1, scale: 1 } : {}}
                transition={{ duration: 0.4, delay: 0.9 + index * 0.1 }}
              >
                <div className="text-3xl font-bold mb-1" style={{ color: stat.color }}>
                  {stat.value}
                </div>
                <div className="font-medium">{stat.label}</div>
                <div className="text-sm text-gray-500">{stat.desc}</div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
