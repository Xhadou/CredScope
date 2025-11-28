import { motion } from 'framer-motion';
import { useInView } from 'framer-motion';
import { useRef } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend } from 'recharts';

const modelComparison = [
  { name: 'LightGBM', auc: 78.52, weight: 35.9 },
  { name: 'XGBoost', auc: 78.31, weight: 32.1 },
  { name: 'CatBoost', auc: 78.28, weight: 32.0 },
  { name: 'Ensemble', auc: 79.08, weight: 100 },
];

const performanceMetrics = [
  { metric: 'AUC', value: 79 },
  { metric: 'Precision', value: 74 },
  { metric: 'Recall', value: 68 },
  { metric: 'F1 Score', value: 71 },
  { metric: 'Accuracy', value: 76 },
];

const radarData = [
  { subject: 'AUC', CredScope: 79, Baseline: 65, fullMark: 100 },
  { subject: 'Precision', CredScope: 74, Baseline: 58, fullMark: 100 },
  { subject: 'Recall', CredScope: 68, Baseline: 52, fullMark: 100 },
  { subject: 'F1', CredScope: 71, Baseline: 55, fullMark: 100 },
  { subject: 'Speed', CredScope: 85, Baseline: 90, fullMark: 100 },
  { subject: 'Explainability', CredScope: 92, Baseline: 30, fullMark: 100 },
];

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card !p-3 text-sm">
        <p className="font-semibold">{label}</p>
        {payload.map((entry, index) => (
          <p key={index} style={{ color: entry.color }}>
            {entry.name}: {entry.value}%
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export default function PerformanceCharts() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  return (
    <section id="performance" className="py-32 relative" ref={ref}>
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-indigo-500/5 via-transparent to-cyan-500/5" />
      
      <div className="max-w-7xl mx-auto px-6 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center mb-20"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            Model <span className="gradient-text">Performance</span>
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Our ensemble approach combines three state-of-the-art gradient boosting models
            for superior predictive performance.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Model Comparison Bar Chart */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="glass-card"
          >
            <h3 className="text-xl font-semibold mb-6">Model AUC Comparison</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={modelComparison} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis type="number" domain={[75, 80]} stroke="#9ca3af" tickFormatter={(v) => `${v}%`} />
                <YAxis dataKey="name" type="category" stroke="#9ca3af" width={80} />
                <Tooltip content={<CustomTooltip />} />
                <Bar 
                  dataKey="auc" 
                  fill="url(#barGradient)"
                  radius={[0, 8, 8, 0]}
                />
                <defs>
                  <linearGradient id="barGradient" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#6366f1" />
                    <stop offset="100%" stopColor="#22d3ee" />
                  </linearGradient>
                </defs>
              </BarChart>
            </ResponsiveContainer>
            <div className="mt-4 flex flex-wrap gap-4 justify-center text-sm">
              {modelComparison.slice(0, 3).map((model) => (
                <div key={model.name} className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-indigo-500" />
                  <span className="text-gray-400">{model.name}: {model.weight}% weight</span>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Radar Chart Comparison */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="glass-card"
          >
            <h3 className="text-xl font-semibold mb-6">CredScope vs Baseline</h3>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="rgba(255,255,255,0.1)" />
                <PolarAngleAxis dataKey="subject" stroke="#9ca3af" />
                <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#9ca3af" />
                <Radar
                  name="CredScope"
                  dataKey="CredScope"
                  stroke="#6366f1"
                  fill="#6366f1"
                  fillOpacity={0.3}
                />
                <Radar
                  name="Baseline"
                  dataKey="Baseline"
                  stroke="#ef4444"
                  fill="#ef4444"
                  fillOpacity={0.1}
                />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </motion.div>
        </div>

        {/* Metrics Cards */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-8"
        >
          {performanceMetrics.map((metric, index) => (
            <motion.div
              key={metric.metric}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={isInView ? { opacity: 1, scale: 1 } : {}}
              transition={{ duration: 0.4, delay: 0.5 + index * 0.1 }}
              whileHover={{ scale: 1.05 }}
              className="glass-card text-center"
            >
              <div className="text-3xl font-bold gradient-text mb-2">{metric.value}%</div>
              <div className="text-gray-400 text-sm">{metric.metric}</div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
