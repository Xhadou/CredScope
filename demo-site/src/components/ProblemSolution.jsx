import { motion } from 'framer-motion';
import { useInView } from 'framer-motion';
import { useRef } from 'react';
import { AlertTriangle, CheckCircle, Zap, Shield, TrendingDown, Eye } from 'lucide-react';

const problems = [
  {
    icon: AlertTriangle,
    title: 'High Default Rates',
    description: 'Traditional scoring misses 30% of risky applicants, leading to significant financial losses.'
  },
  {
    icon: TrendingDown,
    title: 'Limited Data Utilization',
    description: 'Legacy systems only use basic credit bureau data, ignoring rich behavioral signals.'
  },
  {
    icon: Eye,
    title: 'Black Box Decisions',
    description: 'Existing models lack explainability, making regulatory compliance challenging.'
  }
];

const solutions = [
  {
    icon: Shield,
    title: '79% Prediction Accuracy',
    description: 'Ensemble of 3 models achieves 79.08% AUC, significantly outperforming baselines.'
  },
  {
    icon: Zap,
    title: '522 Engineered Features',
    description: 'Advanced feature engineering from 7 data sources captures complex patterns.'
  },
  {
    icon: CheckCircle,
    title: 'Full Transparency',
    description: 'SHAP-based explanations show exactly why each decision was made.'
  }
];

export default function ProblemSolution() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  return (
    <section id="features" className="py-32 relative" ref={ref}>
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center mb-20"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            From Problem to <span className="gradient-text">Solution</span>
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            CredScope addresses the critical challenges in credit risk assessment with cutting-edge ML technology.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-12 items-start">
          {/* Problems */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <div className="flex items-center gap-3 mb-8">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <h3 className="text-2xl font-semibold text-red-400">The Problem</h3>
            </div>
            <div className="space-y-6">
              {problems.map((problem, index) => (
                <motion.div
                  key={problem.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={isInView ? { opacity: 1, y: 0 } : {}}
                  transition={{ duration: 0.5, delay: 0.3 + index * 0.1 }}
                  className="glass-card border-red-500/20 hover:border-red-500/40 transition-colors group"
                >
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 rounded-xl bg-red-500/10 flex items-center justify-center shrink-0 group-hover:bg-red-500/20 transition-colors">
                      <problem.icon className="w-6 h-6 text-red-400" />
                    </div>
                    <div>
                      <h4 className="text-lg font-semibold mb-2">{problem.title}</h4>
                      <p className="text-gray-400">{problem.description}</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Solutions */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <div className="flex items-center gap-3 mb-8">
              <div className="w-3 h-3 rounded-full bg-green-500" />
              <h3 className="text-2xl font-semibold text-green-400">Our Solution</h3>
            </div>
            <div className="space-y-6">
              {solutions.map((solution, index) => (
                <motion.div
                  key={solution.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={isInView ? { opacity: 1, y: 0 } : {}}
                  transition={{ duration: 0.5, delay: 0.3 + index * 0.1 }}
                  className="glass-card border-green-500/20 hover:border-green-500/40 transition-colors group"
                >
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 rounded-xl bg-green-500/10 flex items-center justify-center shrink-0 group-hover:bg-green-500/20 transition-colors">
                      <solution.icon className="w-6 h-6 text-green-400" />
                    </div>
                    <div>
                      <h4 className="text-lg font-semibold mb-2">{solution.title}</h4>
                      <p className="text-gray-400">{solution.description}</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
