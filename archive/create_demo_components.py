"""Script to create React demo site components"""
import os

# Create directory
os.makedirs('demo-site/src/components', exist_ok=True)

# Hero component
hero = '''import { motion, useInView } from 'framer-motion';
import { useRef, useEffect, useState } from 'react';
import { TrendingUp, Layers, Database, Brain, ChevronDown } from 'lucide-react';
import { heroMetrics } from '../data';

const iconMap = { TrendingUp, Layers, Database, Brain };

function AnimatedCounter({ value, suffix = '', duration = 2 }) {
  const [count, setCount] = useState(0);
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true });

  useEffect(() => {
    if (!isInView) return;
    let start = 0;
    const end = value;
    const increment = end / (duration * 60);
    const timer = setInterval(() => {
      start += increment;
      if (start >= end) { setCount(end); clearInterval(timer); }
      else { setCount(start); }
    }, 1000 / 60);
    return () => clearInterval(timer);
  }, [isInView, value, duration]);

  return (
    <span ref={ref}>
      {typeof value === 'number' && value % 1 !== 0 ? count.toFixed(2) : Math.floor(count)}
      {suffix}
    </span>
  );
}

export default function Hero() {
  return (
    <section className="min-h-screen flex flex-col justify-center items-center relative overflow-hidden px-4">
      <div className="absolute inset-0 bg-gradient-to-br from-indigo-950/50 via-zinc-950 to-violet-950/30" />
      <motion.div
        className="absolute top-1/4 left-1/4 w-96 h-96 bg-indigo-500/20 rounded-full blur-3xl"
        animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.5, 0.3] }}
        transition={{ duration: 8, repeat: Infinity }}
      />
      <motion.div
        className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-violet-500/20 rounded-full blur-3xl"
        animate={{ scale: [1.2, 1, 1.2], opacity: [0.5, 0.3, 0.5] }}
        transition={{ duration: 8, repeat: Infinity }}
      />

      <div className="relative z-10 max-w-6xl mx-auto text-center">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass mb-8">
          <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
          <span className="text-zinc-400 text-sm">Machine Learning Credit Risk Assessment</span>
        </motion.div>

        <motion.h1 initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.2 }}
          className="text-5xl md:text-7xl font-bold mb-6 tracking-tight">
          <span className="text-zinc-100">Intelligent</span><br />
          <span className="gradient-text">Credit Decisions</span>
        </motion.h1>

        <motion.p initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.4 }}
          className="text-xl text-zinc-400 max-w-2xl mx-auto mb-12">
          An advanced ensemble machine learning system that predicts loan default risk with state-of-the-art accuracy using 522+ engineered features.
        </motion.p>

        <motion.div initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.6 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6">
          {heroMetrics.map((metric, index) => {
            const Icon = iconMap[metric.icon];
            return (
              <motion.div key={metric.label}
                initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: 0.8 + index * 0.1 }}
                whileHover={{ scale: 1.05, transition: { duration: 0.2 } }}
                className="glass rounded-2xl p-6 glow cursor-default">
                <Icon className="w-8 h-8 text-indigo-400 mx-auto mb-3" />
                <div className="text-3xl md:text-4xl font-bold text-zinc-100 mb-1">
                  <AnimatedCounter value={metric.value} suffix={metric.suffix} />
                </div>
                <div className="text-sm text-zinc-500">{metric.label}</div>
              </motion.div>
            );
          })}
        </motion.div>

        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.5 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2">
          <motion.div animate={{ y: [0, 10, 0] }} transition={{ duration: 2, repeat: Infinity }}
            className="flex flex-col items-center text-zinc-500">
            <span className="text-sm mb-2">Scroll to explore</span>
            <ChevronDown className="w-5 h-5" />
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}'''

# Problem Solution component
problem_solution = '''import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import { AlertTriangle, CheckCircle2, Clock, Ban, Users, Shield, Zap, Target } from 'lucide-react';

const problems = [
  { icon: AlertTriangle, text: 'High default rates erode profitability' },
  { icon: Clock, text: 'Manual review processes are slow and inconsistent' },
  { icon: Ban, text: 'Traditional scoring misses subtle risk patterns' },
  { icon: Users, text: 'Qualified borrowers rejected due to limited data' },
];

const solutions = [
  { icon: Target, text: '79% AUC ensemble model captures complex patterns' },
  { icon: Zap, text: 'Instant predictions with explainable decisions' },
  { icon: Shield, text: '522+ features from 7 diverse data sources' },
  { icon: CheckCircle2, text: 'Fair and transparent risk assessment' },
];

function AnimatedSection({ children, delay = 0 }) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 40 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.8, delay }}
    >
      {children}
    </motion.div>
  );
}

export default function ProblemSolution() {
  return (
    <section className="py-24 px-4">
      <div className="max-w-6xl mx-auto">
        <AnimatedSection>
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-4">
            <span className="text-zinc-100">The Challenge &</span>{' '}
            <span className="gradient-text">Our Solution</span>
          </h2>
          <p className="text-zinc-400 text-center max-w-2xl mx-auto mb-16">
            Traditional credit scoring methods fail to capture the full picture.
            CredScope transforms how lenders assess risk.
          </p>
        </AnimatedSection>

        <div className="grid md:grid-cols-2 gap-8 md:gap-12">
          <AnimatedSection delay={0.2}>
            <div className="glass rounded-2xl p-8">
              <h3 className="text-xl font-semibold text-red-400 mb-6 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5" />
                The Problem
              </h3>
              <div className="space-y-4">
                {problems.map((item, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 + index * 0.1 }}
                    viewport={{ once: true }}
                    className="flex items-start gap-3"
                  >
                    <div className="p-2 rounded-lg bg-red-500/10">
                      <item.icon className="w-5 h-5 text-red-400" />
                    </div>
                    <span className="text-zinc-300">{item.text}</span>
                  </motion.div>
                ))}
              </div>
            </div>
          </AnimatedSection>

          <AnimatedSection delay={0.4}>
            <div className="glass rounded-2xl p-8 glow">
              <h3 className="text-xl font-semibold text-emerald-400 mb-6 flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5" />
                The Solution
              </h3>
              <div className="space-y-4">
                {solutions.map((item, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: 20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.5 + index * 0.1 }}
                    viewport={{ once: true }}
                    className="flex items-start gap-3"
                  >
                    <div className="p-2 rounded-lg bg-emerald-500/10">
                      <item.icon className="w-5 h-5 text-emerald-400" />
                    </div>
                    <span className="text-zinc-300">{item.text}</span>
                  </motion.div>
                ))}
              </div>
            </div>
          </AnimatedSection>
        </div>
      </div>
    </section>
  );
}'''

# Performance component
performance = '''import { motion, useInView } from 'framer-motion';
import { useRef, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { performanceData } from '../data';

const metrics = ['auc', 'precision', 'recall', 'f1'];
const metricLabels = { auc: 'AUC Score', precision: 'Precision', recall: 'Recall', f1: 'F1 Score' };
const colors = { LightGBM: '#818cf8', XGBoost: '#a78bfa', CatBoost: '#c084fc', Ensemble: '#34d399' };

function CustomTooltip({ active, payload, label }) {
  if (active && payload && payload.length) {
    return (
      <div className="glass rounded-lg p-3">
        <p className="text-zinc-100 font-medium">{label}</p>
        <p className="text-indigo-400">{payload[0].value.toFixed(2)}%</p>
      </div>
    );
  }
  return null;
}

export default function Performance() {
  const [activeMetric, setActiveMetric] = useState('auc');
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  const chartData = performanceData.map(d => ({ model: d.model, value: d[activeMetric] }));

  return (
    <section className="py-24 px-4 bg-zinc-900/50" ref={ref}>
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-4">
            <span className="gradient-text">Model Performance</span>
          </h2>
          <p className="text-zinc-400 text-center max-w-2xl mx-auto mb-12">
            Compare the performance of individual models and the ensemble system across key metrics.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="flex justify-center gap-2 mb-8 flex-wrap"
        >
          {metrics.map(metric => (
            <button
              key={metric}
              onClick={() => setActiveMetric(metric)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeMetric === metric ? 'bg-indigo-500 text-white' : 'glass text-zinc-400 hover:text-zinc-100'
              }`}
            >
              {metricLabels[metric]}
            </button>
          ))}
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={isInView ? { opacity: 1, scale: 1 } : {}}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="glass rounded-2xl p-6 glow"
        >
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={chartData} layout="vertical" margin={{ top: 20, right: 30, left: 80, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
              <XAxis type="number" domain={[0, 100]} tick={{ fill: '#a1a1aa' }} axisLine={{ stroke: '#3f3f46' }} />
              <YAxis type="category" dataKey="model" tick={{ fill: '#a1a1aa' }} axisLine={{ stroke: '#3f3f46' }} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="value" radius={[0, 8, 8, 0]}>
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={colors[entry.model]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={isInView ? { opacity: 1 } : {}}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="flex justify-center gap-6 mt-6 flex-wrap"
        >
          {Object.entries(colors).map(([model, color]) => (
            <div key={model} className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-zinc-400 text-sm">{model}</span>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}'''

# Ensemble Architecture component
ensemble = '''import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import { ArrowRight, Sparkles } from 'lucide-react';
import { ensembleWeights } from '../data';

const models = [
  { name: 'LightGBM', color: '#818cf8', weight: ensembleWeights.lightgbm },
  { name: 'XGBoost', color: '#a78bfa', weight: ensembleWeights.xgboost },
  { name: 'CatBoost', color: '#c084fc', weight: ensembleWeights.catboost },
];

export default function EnsembleArchitecture() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  return (
    <section className="py-24 px-4" ref={ref}>
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-4">
            <span className="gradient-text">Ensemble Architecture</span>
          </h2>
          <p className="text-zinc-400 text-center max-w-2xl mx-auto mb-16">
            Three state-of-the-art gradient boosting models combined with optimized weights for superior prediction accuracy.
          </p>
        </motion.div>

        <div className="relative">
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="flex justify-center mb-8"
          >
            <div className="glass rounded-2xl px-8 py-4 inline-flex items-center gap-3">
              <div className="w-3 h-3 bg-indigo-400 rounded-full animate-pulse" />
              <span className="text-zinc-100 font-medium">522 Features Input</span>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={isInView ? { opacity: 1 } : {}}
            transition={{ duration: 0.4, delay: 0.4 }}
            className="flex justify-center mb-8"
          >
            <motion.div animate={{ y: [0, 5, 0] }} transition={{ duration: 1.5, repeat: Infinity }}>
              <ArrowRight className="w-6 h-6 text-indigo-400 rotate-90" />
            </motion.div>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-6 mb-8">
            {models.map((model, index) => (
              <motion.div
                key={model.name}
                initial={{ opacity: 0, y: 30 }}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.6, delay: 0.5 + index * 0.1 }}
                whileHover={{ scale: 1.02 }}
                className="glass rounded-2xl p-6 relative overflow-hidden"
              >
                <div className="absolute top-0 left-0 h-1 w-full" style={{ backgroundColor: model.color }} />
                <h3 className="text-xl font-semibold text-zinc-100 mb-2">{model.name}</h3>
                <p className="text-zinc-400 text-sm mb-4">Gradient Boosting</p>
                <div className="flex items-baseline gap-1">
                  <span className="text-3xl font-bold" style={{ color: model.color }}>
                    {(model.weight * 100).toFixed(1)}%
                  </span>
                  <span className="text-zinc-500 text-sm">weight</span>
                </div>
                <div className="mt-4 h-2 bg-zinc-800 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={isInView ? { width: `${model.weight * 100}%` } : {}}
                    transition={{ duration: 1, delay: 0.8 + index * 0.1 }}
                    className="h-full rounded-full"
                    style={{ backgroundColor: model.color }}
                  />
                </div>
              </motion.div>
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={isInView ? { opacity: 1 } : {}}
            transition={{ duration: 0.4, delay: 0.8 }}
            className="flex justify-center mb-8"
          >
            <div className="flex items-center gap-4">
              <motion.div animate={{ y: [0, 5, 0] }} transition={{ duration: 1.5, repeat: Infinity, delay: 0 }}>
                <ArrowRight className="w-5 h-5 text-indigo-400 rotate-45" />
              </motion.div>
              <motion.div animate={{ y: [0, 5, 0] }} transition={{ duration: 1.5, repeat: Infinity, delay: 0.2 }}>
                <ArrowRight className="w-6 h-6 text-violet-400 rotate-90" />
              </motion.div>
              <motion.div animate={{ y: [0, 5, 0] }} transition={{ duration: 1.5, repeat: Infinity, delay: 0.4 }}>
                <ArrowRight className="w-5 h-5 text-purple-400 rotate-[135deg]" />
              </motion.div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={isInView ? { opacity: 1, scale: 1 } : {}}
            transition={{ duration: 0.6, delay: 1 }}
            className="flex justify-center"
          >
            <div className="glass rounded-2xl px-8 py-6 glow inline-flex items-center gap-4">
              <Sparkles className="w-6 h-6 text-emerald-400" />
              <div>
                <div className="text-zinc-100 font-semibold">Ensemble Prediction</div>
                <div className="text-emerald-400 text-2xl font-bold">79.08% AUC</div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}'''

# Feature Importance component
feature_importance = '''import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { featureImportance } from '../data';

const categoryColors = { External: '#34d399', Client: '#818cf8', Application: '#a78bfa', Bureau: '#fbbf24' };

function CustomTooltip({ active, payload }) {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="glass rounded-lg p-3">
        <p className="text-zinc-100 font-medium">{data.feature}</p>
        <p className="text-indigo-400">Importance: {(data.importance * 100).toFixed(1)}%</p>
        <p className="text-zinc-500 text-sm">Category: {data.category}</p>
      </div>
    );
  }
  return null;
}

export default function FeatureImportance() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  return (
    <section className="py-24 px-4 bg-zinc-900/50" ref={ref}>
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-4">
            <span className="gradient-text">Feature Importance</span>
          </h2>
          <p className="text-zinc-400 text-center max-w-2xl mx-auto mb-12">
            Top predictive features identified by the ensemble model, ranked by their contribution to prediction accuracy.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={isInView ? { opacity: 1, scale: 1 } : {}}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="glass rounded-2xl p-6"
        >
          <ResponsiveContainer width="100%" height={500}>
            <BarChart data={featureImportance} layout="vertical" margin={{ top: 20, right: 30, left: 150, bottom: 20 }}>
              <XAxis type="number" domain={[0, 0.15]} tick={{ fill: '#a1a1aa' }} axisLine={{ stroke: '#3f3f46' }}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
              <YAxis type="category" dataKey="feature" tick={{ fill: '#a1a1aa', fontSize: 12 }} axisLine={{ stroke: '#3f3f46' }} width={140} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="importance" radius={[0, 8, 8, 0]}>
                {featureImportance.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={categoryColors[entry.category]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={isInView ? { opacity: 1 } : {}}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="flex justify-center gap-6 mt-6 flex-wrap"
        >
          {Object.entries(categoryColors).map(([category, color]) => (
            <div key={category} className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-zinc-400 text-sm">{category}</span>
            </div>
          ))}
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="mt-8 glass rounded-xl p-6 text-center"
        >
          <p className="text-zinc-300">
            <span className="text-emerald-400 font-semibold">External credit scores</span>{' '}
            are the strongest predictors, followed by{' '}
            <span className="text-indigo-400 font-semibold">client demographics</span>{' '}
            and employment history.
          </p>
        </motion.div>
      </div>
    </section>
  );
}'''

# Data Sources component
data_sources = '''import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import { FileText, Building2, BarChart3, History, CreditCard, Wallet, Calendar } from 'lucide-react';
import { dataSources } from '../data';

const iconMap = { FileText, Building2, BarChart3, History, CreditCard, Wallet, Calendar };

export default function DataSources() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  return (
    <section className="py-24 px-4" ref={ref}>
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-4">
            <span className="gradient-text">Data Sources</span>
          </h2>
          <p className="text-zinc-400 text-center max-w-2xl mx-auto mb-12">
            Comprehensive feature engineering from 7 distinct data sources, creating a holistic view of each applicant.
          </p>
        </motion.div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={isInView ? { opacity: 1, scale: 1 } : {}}
            transition={{ duration: 0.6, delay: 0.2 }}
            whileHover={{ scale: 1.02 }}
            className="col-span-2 row-span-2 glass rounded-2xl p-6 flex flex-col justify-between"
          >
            <div>
              <div className="w-12 h-12 rounded-xl bg-indigo-500/10 flex items-center justify-center mb-4">
                <FileText className="w-6 h-6 text-indigo-400" />
              </div>
              <h3 className="text-xl font-semibold text-zinc-100 mb-2">Application Data</h3>
              <p className="text-zinc-400 text-sm">Core loan application information including demographics, income, and requested amounts.</p>
            </div>
            <div className="mt-4">
              <span className="text-3xl font-bold text-indigo-400">122</span>
              <span className="text-zinc-500 ml-2">features</span>
            </div>
          </motion.div>

          {dataSources.slice(1).map((source, index) => {
            const Icon = iconMap[source.icon];
            const colorClasses = {
              indigo: { bg: 'bg-indigo-500/10', text: 'text-indigo-400' },
              violet: { bg: 'bg-violet-500/10', text: 'text-violet-400' },
              emerald: { bg: 'bg-emerald-500/10', text: 'text-emerald-400' },
              amber: { bg: 'bg-amber-500/10', text: 'text-amber-400' },
            };
            const colors = colorClasses[source.color];
            return (
              <motion.div
                key={source.name}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={isInView ? { opacity: 1, scale: 1 } : {}}
                transition={{ duration: 0.5, delay: 0.3 + index * 0.1 }}
                whileHover={{ scale: 1.05 }}
                className="glass rounded-2xl p-4 flex flex-col"
              >
                <div className={`w-10 h-10 rounded-lg ${colors.bg} flex items-center justify-center mb-3`}>
                  <Icon className={`w-5 h-5 ${colors.text}`} />
                </div>
                <h3 className="text-sm font-semibold text-zinc-100 mb-1">{source.name}</h3>
                <p className="text-xs text-zinc-500 flex-grow">{source.description}</p>
                <div className="mt-3">
                  <span className={`text-xl font-bold ${colors.text}`}>{source.features}</span>
                  <span className="text-zinc-600 text-xs ml-1">features</span>
                </div>
              </motion.div>
            );
          })}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.8 }}
          className="mt-6 glass rounded-xl p-6 text-center glow"
        >
          <div className="text-4xl font-bold gradient-text mb-2">522+</div>
          <div className="text-zinc-400">Total Engineered Features</div>
        </motion.div>
      </div>
    </section>
  );
}'''

# Live Demo component
live_demo = '''import { motion, useInView } from 'framer-motion';
import { useRef, useState, useMemo } from 'react';
import { Play, RotateCcw, CheckCircle2, AlertTriangle, XCircle } from 'lucide-react';

const defaultInputs = { income: 150000, credit: 500000, age: 35, employment: 5, extScore: 0.65 };

function calculateRisk(inputs) {
  const incomeRatio = inputs.credit / inputs.income;
  const ageScore = inputs.age > 25 && inputs.age < 55 ? 0.8 : 0.5;
  const empScore = Math.min(inputs.employment / 10, 1);
  const extScore = inputs.extScore;
  const baseRisk = (1 - extScore) * 40 + (incomeRatio / 10) * 20 + (1 - ageScore) * 15 + (1 - empScore) * 10;
  return Math.max(5, Math.min(85, baseRisk + Math.random() * 10 - 5));
}

function RiskGauge({ risk }) {
  const getColor = (r) => r < 20 ? '#34d399' : r < 50 ? '#fbbf24' : '#f87171';
  const rotation = (risk / 100) * 180 - 90;

  return (
    <div className="relative w-48 h-24 mx-auto">
      <svg className="w-full h-full" viewBox="0 0 200 100">
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#3f3f46" strokeWidth="16" strokeLinecap="round" />
        <path d="M 20 100 A 80 80 0 0 1 56 36" fill="none" stroke="#34d399" strokeWidth="16" strokeLinecap="round" opacity="0.3" />
        <path d="M 56 36 A 80 80 0 0 1 144 36" fill="none" stroke="#fbbf24" strokeWidth="16" opacity="0.3" />
        <path d="M 144 36 A 80 80 0 0 1 180 100" fill="none" stroke="#f87171" strokeWidth="16" strokeLinecap="round" opacity="0.3" />
      </svg>
      <motion.div
        className="absolute bottom-0 left-1/2 origin-bottom"
        initial={{ rotate: -90 }}
        animate={{ rotate: rotation }}
        transition={{ type: 'spring', stiffness: 60, damping: 15 }}
        style={{ width: '4px', height: '70px', marginLeft: '-2px' }}
      >
        <div className="w-full h-full rounded-full" style={{ backgroundColor: getColor(risk) }} />
      </motion.div>
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-4 h-4 rounded-full bg-zinc-700 border-2" style={{ borderColor: getColor(risk) }} />
    </div>
  );
}

export default function LiveDemo() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });
  const [inputs, setInputs] = useState(defaultInputs);
  const [isCalculating, setIsCalculating] = useState(false);
  const [risk, setRisk] = useState(null);

  const handleCalculate = () => {
    setIsCalculating(true);
    setRisk(null);
    setTimeout(() => { setRisk(calculateRisk(inputs)); setIsCalculating(false); }, 1500);
  };

  const handleReset = () => { setInputs(defaultInputs); setRisk(null); };

  const decision = useMemo(() => {
    if (risk === null) return null;
    if (risk < 20) return { label: 'Approve', icon: CheckCircle2, color: 'emerald' };
    if (risk < 50) return { label: 'Review', icon: AlertTriangle, color: 'amber' };
    return { label: 'Reject', icon: XCircle, color: 'red' };
  }, [risk]);

  const decisionColors = { emerald: 'bg-emerald-500/20 text-emerald-400', amber: 'bg-amber-500/20 text-amber-400', red: 'bg-red-500/20 text-red-400' };

  return (
    <section className="py-24 px-4 bg-zinc-900/50" ref={ref}>
      <div className="max-w-4xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 40 }} animate={isInView ? { opacity: 1, y: 0 } : {}} transition={{ duration: 0.8 }}>
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-4">
            <span className="gradient-text">Live Demo</span>
          </h2>
          <p className="text-zinc-400 text-center max-w-2xl mx-auto mb-12">
            Try the risk assessment simulator. Adjust the inputs and see how the ensemble model evaluates credit risk.
          </p>
        </motion.div>

        <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={isInView ? { opacity: 1, scale: 1 } : {}} transition={{ duration: 0.8, delay: 0.2 }}
          className="glass rounded-2xl p-8 glow">
          <div className="grid md:grid-cols-2 gap-6 mb-8">
            <div>
              <label className="text-zinc-400 text-sm mb-2 block">Annual Income</label>
              <input type="range" min="30000" max="500000" step="10000" value={inputs.income}
                onChange={(e) => setInputs({ ...inputs, income: parseInt(e.target.value) })} className="w-full accent-indigo-500" />
              <div className="text-indigo-400 font-semibold">${inputs.income.toLocaleString()}</div>
            </div>
            <div>
              <label className="text-zinc-400 text-sm mb-2 block">Loan Amount</label>
              <input type="range" min="50000" max="2000000" step="50000" value={inputs.credit}
                onChange={(e) => setInputs({ ...inputs, credit: parseInt(e.target.value) })} className="w-full accent-indigo-500" />
              <div className="text-indigo-400 font-semibold">${inputs.credit.toLocaleString()}</div>
            </div>
            <div>
              <label className="text-zinc-400 text-sm mb-2 block">Age (years)</label>
              <input type="range" min="21" max="65" value={inputs.age}
                onChange={(e) => setInputs({ ...inputs, age: parseInt(e.target.value) })} className="w-full accent-indigo-500" />
              <div className="text-indigo-400 font-semibold">{inputs.age} years</div>
            </div>
            <div>
              <label className="text-zinc-400 text-sm mb-2 block">Employment (years)</label>
              <input type="range" min="0" max="30" value={inputs.employment}
                onChange={(e) => setInputs({ ...inputs, employment: parseInt(e.target.value) })} className="w-full accent-indigo-500" />
              <div className="text-indigo-400 font-semibold">{inputs.employment} years</div>
            </div>
            <div className="md:col-span-2">
              <label className="text-zinc-400 text-sm mb-2 block">External Credit Score (0-1)</label>
              <input type="range" min="0" max="1" step="0.01" value={inputs.extScore}
                onChange={(e) => setInputs({ ...inputs, extScore: parseFloat(e.target.value) })} className="w-full accent-indigo-500" />
              <div className="text-indigo-400 font-semibold">{inputs.extScore.toFixed(2)}</div>
            </div>
          </div>

          <div className="flex justify-center gap-4 mb-8">
            <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={handleCalculate} disabled={isCalculating}
              className="px-6 py-3 bg-indigo-500 hover:bg-indigo-600 text-white rounded-lg font-medium flex items-center gap-2 disabled:opacity-50">
              <Play className="w-4 h-4" />
              {isCalculating ? 'Calculating...' : 'Calculate Risk'}
            </motion.button>
            <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={handleReset}
              className="px-6 py-3 glass text-zinc-300 rounded-lg font-medium flex items-center gap-2">
              <RotateCcw className="w-4 h-4" />
              Reset
            </motion.button>
          </div>

          {(risk !== null || isCalculating) && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center">
              {isCalculating ? (
                <div className="flex flex-col items-center gap-4">
                  <div className="w-12 h-12 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                  <span className="text-zinc-400">Running ensemble prediction...</span>
                </div>
              ) : (
                <>
                  <RiskGauge risk={risk} />
                  <div className="mt-4 text-3xl font-bold text-zinc-100">{risk.toFixed(1)}% Risk</div>
                  {decision && (
                    <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: 'spring', delay: 0.2 }}
                      className={`inline-flex items-center gap-2 mt-4 px-4 py-2 rounded-full ${decisionColors[decision.color]}`}>
                      <decision.icon className="w-5 h-5" />
                      <span className="font-semibold">{decision.label}</span>
                    </motion.div>
                  )}
                </>
              )}
            </motion.div>
          )}
        </motion.div>
      </div>
    </section>
  );
}'''

# Thresholds component
thresholds = '''import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import { CheckCircle2, AlertTriangle, XCircle, ArrowRight } from 'lucide-react';

const zones = [
  { range: '0% - 20%', label: 'Auto Approve', description: 'Low risk applicants processed automatically', icon: CheckCircle2, color: 'emerald' },
  { range: '20% - 50%', label: 'Manual Review', description: 'Moderate risk requires human assessment', icon: AlertTriangle, color: 'amber' },
  { range: '50% - 100%', label: 'Auto Reject', description: 'High risk applications declined', icon: XCircle, color: 'red' },
];

export default function Thresholds() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  const colorClasses = {
    emerald: { icon: 'text-emerald-400', value: 'text-emerald-400', bg: 'from-emerald-500/20 to-emerald-500/5' },
    amber: { icon: 'text-amber-400', value: 'text-amber-400', bg: 'from-amber-500/20 to-amber-500/5' },
    red: { icon: 'text-red-400', value: 'text-red-400', bg: 'from-red-500/20 to-red-500/5' },
  };

  return (
    <section className="py-24 px-4" ref={ref}>
      <div className="max-w-6xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 40 }} animate={isInView ? { opacity: 1, y: 0 } : {}} transition={{ duration: 0.8 }}>
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-4">
            <span className="gradient-text">Decision Thresholds</span>
          </h2>
          <p className="text-zinc-400 text-center max-w-2xl mx-auto mb-12">
            Configurable risk thresholds enable automated decisioning while ensuring human oversight where needed.
          </p>
        </motion.div>

        <motion.div initial={{ opacity: 0, scaleX: 0 }} animate={isInView ? { opacity: 1, scaleX: 1 } : {}} transition={{ duration: 1, delay: 0.2 }}
          className="h-4 rounded-full overflow-hidden flex mb-8">
          <div className="w-[20%] bg-emerald-500/60" />
          <div className="w-[30%] bg-amber-500/60" />
          <div className="w-[50%] bg-red-500/60" />
        </motion.div>

        <div className="flex justify-between text-sm text-zinc-500 mb-12 px-2">
          <span>0%</span>
          <span className="-translate-x-4">20%</span>
          <span className="translate-x-2">50%</span>
          <span>100%</span>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {zones.map((zone, index) => {
            const colors = colorClasses[zone.color];
            return (
              <motion.div key={zone.label} initial={{ opacity: 0, y: 30 }} animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.6, delay: 0.4 + index * 0.1 }}
                className={`glass rounded-2xl p-6 bg-gradient-to-b ${colors.bg}`}>
                <zone.icon className={`w-10 h-10 ${colors.icon} mb-4`} />
                <div className={`text-2xl font-bold ${colors.value} mb-2`}>{zone.range}</div>
                <h3 className="text-xl font-semibold text-zinc-100 mb-2">{zone.label}</h3>
                <p className="text-zinc-400 text-sm">{zone.description}</p>
              </motion.div>
            );
          })}
        </div>

        <motion.div initial={{ opacity: 0 }} animate={isInView ? { opacity: 1 } : {}} transition={{ duration: 0.6, delay: 0.8 }}
          className="flex items-center justify-center gap-4 mt-8 text-zinc-500">
          <span className="text-sm">Risk Score</span>
          <ArrowRight className="w-4 h-4" />
          <span className="text-sm">Threshold Check</span>
          <ArrowRight className="w-4 h-4" />
          <span className="text-sm">Decision</span>
        </motion.div>
      </div>
    </section>
  );
}'''

# Fairness component
fairness = '''import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import { Scale, Eye, ShieldCheck, Users } from 'lucide-react';

const principles = [
  { icon: Scale, title: 'Fair Lending', description: 'Model performance monitored across demographic groups to ensure equitable outcomes.', color: 'indigo' },
  { icon: Eye, title: 'Transparency', description: 'SHAP-based explanations provide clear reasoning for every credit decision.', color: 'violet' },
  { icon: ShieldCheck, title: 'Regulatory Compliance', description: 'Designed to meet ECOA, FCRA, and adverse action notice requirements.', color: 'emerald' },
  { icon: Users, title: 'Human Oversight', description: 'Borderline cases routed to human reviewers for final determination.', color: 'amber' },
];

export default function Fairness() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  const colorClasses = {
    indigo: { bg: 'bg-indigo-500/10', icon: 'text-indigo-400' },
    violet: { bg: 'bg-violet-500/10', icon: 'text-violet-400' },
    emerald: { bg: 'bg-emerald-500/10', icon: 'text-emerald-400' },
    amber: { bg: 'bg-amber-500/10', icon: 'text-amber-400' },
  };

  return (
    <section className="py-24 px-4 bg-zinc-900/50" ref={ref}>
      <div className="max-w-6xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 40 }} animate={isInView ? { opacity: 1, y: 0 } : {}} transition={{ duration: 0.8 }}>
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-4">
            <span className="gradient-text">Fairness & Ethics</span>
          </h2>
          <p className="text-zinc-400 text-center max-w-2xl mx-auto mb-12">
            Responsible AI practices are built into every layer of the system, ensuring fair and explainable credit decisions.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-6">
          {principles.map((principle, index) => {
            const colors = colorClasses[principle.color];
            return (
              <motion.div key={principle.title} initial={{ opacity: 0, x: index % 2 === 0 ? -30 : 30 }}
                animate={isInView ? { opacity: 1, x: 0 } : {}} transition={{ duration: 0.6, delay: 0.2 + index * 0.1 }}
                whileHover={{ scale: 1.02 }} className="glass rounded-2xl p-6 flex gap-4">
                <div className={`w-12 h-12 rounded-xl ${colors.bg} flex items-center justify-center flex-shrink-0`}>
                  <principle.icon className={`w-6 h-6 ${colors.icon}`} />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-zinc-100 mb-2">{principle.title}</h3>
                  <p className="text-zinc-400 text-sm">{principle.description}</p>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}'''

# Footer component
footer = '''import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import { Github, ExternalLink } from 'lucide-react';
import { techStack } from '../data';

export default function Footer() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-50px' });

  const techColors = {
    'ML Framework': 'text-indigo-400', Language: 'text-violet-400', Dashboard: 'text-emerald-400',
    API: 'text-amber-400', Deployment: 'text-red-400', Tracking: 'text-indigo-400',
  };

  return (
    <footer className="py-16 px-4 border-t border-zinc-800" ref={ref}>
      <div className="max-w-6xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={isInView ? { opacity: 1, y: 0 } : {}} transition={{ duration: 0.8 }}>
          <h2 className="text-2xl font-bold text-center mb-8">
            <span className="gradient-text">Technology Stack</span>
          </h2>

          <div className="flex flex-wrap justify-center gap-3 mb-12">
            {techStack.map((tech, index) => (
              <motion.div key={tech.name} initial={{ opacity: 0, scale: 0.8 }} animate={isInView ? { opacity: 1, scale: 1 } : {}}
                transition={{ duration: 0.4, delay: 0.1 + index * 0.05 }} whileHover={{ scale: 1.05 }}
                className="px-4 py-2 rounded-full glass text-sm">
                <span className={techColors[tech.type]}>{tech.name}</span>
                <span className="text-zinc-600 ml-2">{tech.type}</span>
              </motion.div>
            ))}
          </div>

          <div className="flex justify-center gap-6 mb-8">
            <motion.a href="#" whileHover={{ scale: 1.05 }} className="flex items-center gap-2 text-zinc-400 hover:text-zinc-100 transition-colors">
              <Github className="w-5 h-5" />
              <span>GitHub</span>
            </motion.a>
            <motion.a href="#" whileHover={{ scale: 1.05 }} className="flex items-center gap-2 text-zinc-400 hover:text-zinc-100 transition-colors">
              <ExternalLink className="w-5 h-5" />
              <span>Documentation</span>
            </motion.a>
          </div>

          <div className="text-center text-zinc-600 text-sm">
            <p>CredScope - Machine Learning Credit Risk Assessment</p>
            <p className="mt-1">Built with LightGBM, XGBoost, CatBoost & React</p>
          </div>
        </motion.div>
      </div>
    </footer>
  );
}'''

# Write all files
with open('demo-site/src/components/Hero.jsx', 'w', encoding='utf-8') as f:
    f.write(hero)
print('Hero.jsx created')

with open('demo-site/src/components/ProblemSolution.jsx', 'w', encoding='utf-8') as f:
    f.write(problem_solution)
print('ProblemSolution.jsx created')

with open('demo-site/src/components/Performance.jsx', 'w', encoding='utf-8') as f:
    f.write(performance)
print('Performance.jsx created')

with open('demo-site/src/components/EnsembleArchitecture.jsx', 'w', encoding='utf-8') as f:
    f.write(ensemble)
print('EnsembleArchitecture.jsx created')

with open('demo-site/src/components/FeatureImportance.jsx', 'w', encoding='utf-8') as f:
    f.write(feature_importance)
print('FeatureImportance.jsx created')

with open('demo-site/src/components/DataSources.jsx', 'w', encoding='utf-8') as f:
    f.write(data_sources)
print('DataSources.jsx created')

with open('demo-site/src/components/LiveDemo.jsx', 'w', encoding='utf-8') as f:
    f.write(live_demo)
print('LiveDemo.jsx created')

with open('demo-site/src/components/Thresholds.jsx', 'w', encoding='utf-8') as f:
    f.write(thresholds)
print('Thresholds.jsx created')

with open('demo-site/src/components/Fairness.jsx', 'w', encoding='utf-8') as f:
    f.write(fairness)
print('Fairness.jsx created')

with open('demo-site/src/components/Footer.jsx', 'w', encoding='utf-8') as f:
    f.write(footer)
print('Footer.jsx created')

print('All components created successfully!')
