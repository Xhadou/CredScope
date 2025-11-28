import { motion, AnimatePresence } from 'framer-motion';
import { useInView } from 'framer-motion';
import { useRef, useState, useEffect, useMemo } from 'react';
import { Play, RotateCcw, CheckCircle, AlertTriangle, XCircle, User, Briefcase, CreditCard, Calendar, DollarSign, TrendingUp } from 'lucide-react';

const sampleApplicants = [
  {
    id: 1,
    name: 'Sarah Chen',
    age: 35,
    income: 85000,
    employment: 'Software Engineer',
    employmentYears: 8,
    creditAmount: 250000,
    extSource1: 0.72,
    extSource2: 0.68,
    extSource3: 0.75,
    riskScore: 0.12,
    decision: 'APPROVE'
  },
  {
    id: 2,
    name: 'Rajesh Patel',
    age: 42,
    income: 120000,
    employment: 'Senior Manager',
    employmentYears: 15,
    creditAmount: 180000,
    extSource1: 0.81,
    extSource2: 0.79,
    extSource3: 0.77,
    riskScore: 0.08,
    decision: 'APPROVE'
  },
  {
    id: 3,
    name: 'Emily Watson',
    age: 29,
    income: 62000,
    employment: 'Accountant',
    employmentYears: 5,
    creditAmount: 150000,
    extSource1: 0.65,
    extSource2: 0.61,
    extSource3: 0.58,
    riskScore: 0.18,
    decision: 'APPROVE'
  },
  {
    id: 4,
    name: 'Marcus Johnson',
    age: 28,
    income: 45000,
    employment: 'Sales Associate',
    employmentYears: 2,
    creditAmount: 150000,
    extSource1: 0.45,
    extSource2: 0.52,
    extSource3: 0.48,
    riskScore: 0.38,
    decision: 'REVIEW'
  },
  {
    id: 5,
    name: 'Anita Sharma',
    age: 31,
    income: 55000,
    employment: 'Marketing Executive',
    employmentYears: 3,
    creditAmount: 200000,
    extSource1: 0.51,
    extSource2: 0.47,
    extSource3: 0.53,
    riskScore: 0.42,
    decision: 'REVIEW'
  },
  {
    id: 6,
    name: 'David Kim',
    age: 26,
    income: 38000,
    employment: 'Junior Developer',
    employmentYears: 1,
    creditAmount: 120000,
    extSource1: 0.42,
    extSource2: 0.55,
    extSource3: 0.44,
    riskScore: 0.35,
    decision: 'REVIEW'
  },
  {
    id: 7,
    name: 'Michael Torres',
    age: 34,
    income: 52000,
    employment: 'Contractor',
    employmentYears: 1,
    creditAmount: 280000,
    extSource1: 0.38,
    extSource2: 0.41,
    extSource3: 0.35,
    riskScore: 0.48,
    decision: 'REVIEW'
  },
  {
    id: 8,
    name: 'Elena Rodriguez',
    age: 42,
    income: 32000,
    employment: 'Part-time Worker',
    employmentYears: 1,
    creditAmount: 200000,
    extSource1: 0.28,
    extSource2: 0.31,
    extSource3: 0.25,
    riskScore: 0.67,
    decision: 'REJECT'
  },
  {
    id: 9,
    name: 'James Wright',
    age: 24,
    income: 22000,
    employment: 'Unemployed',
    employmentYears: 0,
    creditAmount: 150000,
    extSource1: 0.18,
    extSource2: 0.22,
    extSource3: 0.15,
    riskScore: 0.82,
    decision: 'REJECT'
  },
  {
    id: 10,
    name: 'Lisa Brown',
    age: 38,
    income: 28000,
    employment: 'Gig Worker',
    employmentYears: 0,
    creditAmount: 250000,
    extSource1: 0.25,
    extSource2: 0.19,
    extSource3: 0.21,
    riskScore: 0.74,
    decision: 'REJECT'
  },
];

const getDecisionStyle = (decision) => {
  switch (decision) {
    case 'APPROVE':
      return { bg: 'bg-green-500/20', border: 'border-green-500/50', text: 'text-green-400', icon: CheckCircle };
    case 'REVIEW':
      return { bg: 'bg-amber-500/20', border: 'border-amber-500/50', text: 'text-amber-400', icon: AlertTriangle };
    case 'REJECT':
      return { bg: 'bg-red-500/20', border: 'border-red-500/50', text: 'text-red-400', icon: XCircle };
    default:
      return { bg: 'bg-gray-500/20', border: 'border-gray-500/50', text: 'text-gray-400', icon: AlertTriangle };
  }
};

export default function LiveDemo() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });
  const [currentApplicant, setCurrentApplicant] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const [progress, setProgress] = useState(0);

  const runDemo = () => {
    setIsProcessing(true);
    setShowResult(false);
    setProgress(0);
    
    const duration = 6000;
    const interval = 50;
    const steps = duration / interval;
    let step = 0;
    
    const timer = setInterval(() => {
      step++;
      setProgress((step / steps) * 100);
      
      if (step >= steps) {
        clearInterval(timer);
        setIsProcessing(false);
        setShowResult(true);
      }
    }, interval);
  };

  const reset = () => {
    setShowResult(false);
    setProgress(0);
    setCurrentApplicant((prev) => (prev + 1) % shuffledApplicants.length);
  };

  // Shuffle applicants once on component mount
  const shuffledApplicants = useMemo(() => {
    return [...sampleApplicants].sort(() => Math.random() - 0.5);
  }, []);

  const applicant = shuffledApplicants[currentApplicant];
  const decisionStyle = getDecisionStyle(applicant.decision);
  const DecisionIcon = decisionStyle.icon;

  return (
    <section id="demo" className="py-32 relative" ref={ref}>
      <div className="absolute inset-0 bg-gradient-to-b from-cyan-500/5 via-transparent to-indigo-500/5" />
      
      <div className="max-w-7xl mx-auto px-6 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center mb-20"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            Live <span className="gradient-text">Demo</span>
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            See how CredScope evaluates credit applications in real-time with transparent scoring.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="max-w-4xl mx-auto"
        >
          <div className="glass-card">
            {/* Applicant Header */}
            <div className="flex items-center justify-between mb-8 pb-6 border-b border-white/10">
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-indigo-500 to-cyan-500 flex items-center justify-center text-2xl font-bold">
                  {applicant.name.charAt(0)}
                </div>
                <div>
                  <h3 className="text-xl font-semibold">{applicant.name}</h3>
                  <p className="text-gray-400">{applicant.employment} • {applicant.employmentYears} years</p>
                </div>
              </div>
              
              <div className="flex gap-2">
                {!isProcessing && !showResult && (
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={runDemo}
                    className="px-6 py-3 bg-gradient-to-r from-indigo-500 to-cyan-500 rounded-xl font-medium flex items-center gap-2"
                  >
                    <Play className="w-5 h-5" />
                    <span>Analyze</span>
                  </motion.button>
                )}
                {showResult && (
                  <motion.button
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={reset}
                    className="px-6 py-3 glass rounded-xl font-medium flex items-center gap-2"
                  >
                    <RotateCcw className="w-5 h-5" />
                    <span>Next Applicant</span>
                  </motion.button>
                )}
              </div>
            </div>

            {/* Applicant Details Grid */}
            <div className="grid md:grid-cols-2 gap-6 mb-8">
              <div className="space-y-4">
                <div className="flex items-center gap-3 text-gray-400">
                  <User className="w-5 h-5" />
                  <span>Age: <span className="text-white">{applicant.age} years</span></span>
                </div>
                <div className="flex items-center gap-3 text-gray-400">
                  <Briefcase className="w-5 h-5" />
                  <span>Employment: <span className="text-white">{applicant.employment}</span></span>
                </div>
                <div className="flex items-center gap-3 text-gray-400">
                  <Calendar className="w-5 h-5" />
                  <span>Years Employed: <span className={applicant.employmentYears >= 3 ? "text-green-400" : applicant.employmentYears >= 1 ? "text-amber-400" : "text-red-400"}>{applicant.employmentYears} {applicant.employmentYears === 1 ? 'year' : 'years'}</span></span>
                </div>
                <div className="flex items-center gap-3 text-gray-400">
                  <DollarSign className="w-5 h-5" />
                  <span>Annual Income: <span className="text-white">${applicant.income.toLocaleString()}</span></span>
                </div>
                <div className="flex items-center gap-3 text-gray-400">
                  <CreditCard className="w-5 h-5" />
                  <span>Credit Amount: <span className="text-white">${applicant.creditAmount.toLocaleString()}</span></span>
                </div>
                <div className="flex items-center gap-3 text-gray-400">
                  <TrendingUp className="w-5 h-5" />
                  <span>Debt-to-Income: <span className={(applicant.creditAmount / applicant.income) <= 3 ? "text-green-400" : (applicant.creditAmount / applicant.income) <= 5 ? "text-amber-400" : "text-red-400"}>{(applicant.creditAmount / applicant.income).toFixed(1)}x</span></span>
                </div>
              </div>
              
              <div className="space-y-4">
                <h4 className="font-medium text-gray-300">External Credit Scores <span className="text-xs text-gray-500">(most important features)</span></h4>
                {[
                  { name: 'EXT_SOURCE_1', value: applicant.extSource1 },
                  { name: 'EXT_SOURCE_2', value: applicant.extSource2 },
                  { name: 'EXT_SOURCE_3', value: applicant.extSource3 },
                ].map((score) => {
                  const scoreColor = score.value >= 0.6 ? 'from-green-500 to-green-400' : score.value >= 0.4 ? 'from-amber-500 to-amber-400' : 'from-red-500 to-red-400';
                  const textColor = score.value >= 0.6 ? 'text-green-400' : score.value >= 0.4 ? 'text-amber-400' : 'text-red-400';
                  return (
                    <div key={score.name}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-400">{score.name}</span>
                        <span className={textColor}>{(score.value * 100).toFixed(0)}%</span>
                      </div>
                      <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                        <div 
                          className={`h-full rounded-full bg-gradient-to-r ${scoreColor}`}
                          style={{ width: `${score.value * 100}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
                <div className="pt-2 mt-2 border-t border-white/10">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Average Score</span>
                    <span className={((applicant.extSource1 + applicant.extSource2 + applicant.extSource3) / 3) >= 0.6 ? 'text-green-400 font-medium' : ((applicant.extSource1 + applicant.extSource2 + applicant.extSource3) / 3) >= 0.4 ? 'text-amber-400 font-medium' : 'text-red-400 font-medium'}>
                      {(((applicant.extSource1 + applicant.extSource2 + applicant.extSource3) / 3) * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Processing Animation */}
            <AnimatePresence>
              {isProcessing && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mb-8"
                >
                  <div className="bg-white/5 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-gray-400">Processing with ensemble models...</span>
                      <span className="font-mono">{progress.toFixed(0)}%</span>
                    </div>
                    <div className="h-3 rounded-full bg-white/10 overflow-hidden">
                      <motion.div
                        className="h-full rounded-full bg-gradient-to-r from-indigo-500 via-cyan-500 to-indigo-500"
                        style={{ width: `${progress}%` }}
                        animate={{
                          backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
                        }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                        }}
                      />
                    </div>
                    <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
                      {['LightGBM', 'XGBoost', 'CatBoost'].map((model, i) => (
                        <div key={model} className="text-center">
                          <div className={`w-2 h-2 rounded-full mx-auto mb-2 ${progress > (i + 1) * 30 ? 'bg-green-500' : 'bg-white/20'}`} />
                          <span className="text-gray-400">{model}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Result */}
            <AnimatePresence>
              {showResult && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="space-y-6"
                >
                  {/* Risk Score Gauge */}
                  <div className="bg-white/5 rounded-xl p-6">
                    <div className="text-center mb-6">
                      <h4 className="text-gray-400 mb-2">Risk Score</h4>
                      <div className="text-5xl font-bold gradient-text">
                        {(applicant.riskScore * 100).toFixed(1)}%
                      </div>
                    </div>
                    
                    {/* Gauge bar */}
                    <div className="relative h-4 rounded-full overflow-hidden">
                      <div className="absolute inset-0 flex">
                        <div className="w-1/5 bg-green-500/30" />
                        <div className="w-3/10 bg-amber-500/30" style={{ width: '30%' }} />
                        <div className="flex-1 bg-red-500/30" />
                      </div>
                      <motion.div
                        initial={{ left: 0 }}
                        animate={{ left: `${applicant.riskScore * 100}%` }}
                        transition={{ duration: 0.5, delay: 0.2 }}
                        className="absolute top-0 w-1 h-full bg-white rounded-full shadow-lg shadow-white/50"
                        style={{ transform: 'translateX(-50%)' }}
                      />
                    </div>
                    
                    <div className="flex justify-between text-sm mt-2 text-gray-500">
                      <span>0% - Low Risk</span>
                      <span>20%</span>
                      <span>50%</span>
                      <span>100% - High Risk</span>
                    </div>
                  </div>

                  {/* Decision */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className={`p-6 rounded-xl border ${decisionStyle.bg} ${decisionStyle.border}`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <DecisionIcon className={`w-8 h-8 ${decisionStyle.text}`} />
                        <div>
                          <div className={`text-2xl font-bold ${decisionStyle.text}`}>
                            {applicant.decision}
                          </div>
                          <div className="text-gray-400 text-sm mt-1">
                            {applicant.decision === 'APPROVE' && (
                              <>
                                Strong profile: {((applicant.extSource1 + applicant.extSource2 + applicant.extSource3) / 3 * 100).toFixed(0)}% avg credit score, 
                                {applicant.employmentYears}+ years employment, 
                                {(applicant.creditAmount / applicant.income).toFixed(1)}x DTI ratio
                              </>
                            )}
                            {applicant.decision === 'REVIEW' && (
                              <>
                                Mixed signals: {((applicant.extSource1 + applicant.extSource2 + applicant.extSource3) / 3 * 100).toFixed(0)}% avg credit score, 
                                {applicant.employmentYears <= 2 ? 'limited employment history' : `${applicant.employmentYears} years employed`}, 
                                {(applicant.creditAmount / applicant.income).toFixed(1)}x DTI ratio
                              </>
                            )}
                            {applicant.decision === 'REJECT' && (
                              <>
                                High risk: {((applicant.extSource1 + applicant.extSource2 + applicant.extSource3) / 3 * 100).toFixed(0)}% avg credit score, 
                                {applicant.employmentYears === 0 ? 'no stable employment' : `only ${applicant.employmentYears} year employed`}, 
                                {(applicant.creditAmount / applicant.income).toFixed(1)}x DTI ratio
                              </>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
