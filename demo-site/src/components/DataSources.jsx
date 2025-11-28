import { motion } from 'framer-motion';
import { useInView } from 'framer-motion';
import { useRef } from 'react';
import { FileSpreadsheet, Building2, History, CreditCard, Banknote, Calculator, Database } from 'lucide-react';

const dataSources = [
  {
    name: 'application_train/test',
    icon: FileSpreadsheet,
    records: '307K / 48K',
    features: '122',
    description: 'Main application data with target',
    color: '#6366f1',
    size: 'large'
  },
  {
    name: 'bureau',
    icon: Building2,
    records: '1.7M',
    features: '17',
    description: 'Credit bureau loan history',
    color: '#22c55e',
    size: 'medium'
  },
  {
    name: 'bureau_balance',
    icon: History,
    records: '27.3M',
    features: '3',
    description: 'Monthly bureau balance snapshots',
    color: '#f59e0b',
    size: 'medium'
  },
  {
    name: 'previous_application',
    icon: Database,
    records: '1.7M',
    features: '37',
    description: 'Previous Home Credit applications',
    color: '#ec4899',
    size: 'medium'
  },
  {
    name: 'POS_CASH_balance',
    icon: Banknote,
    records: '10M',
    features: '8',
    description: 'POS and cash loan snapshots',
    color: '#14b8a6',
    size: 'small'
  },
  {
    name: 'installments_payments',
    icon: Calculator,
    records: '13.6M',
    features: '8',
    description: 'Payment history for loans',
    color: '#8b5cf6',
    size: 'small'
  },
  {
    name: 'credit_card_balance',
    icon: CreditCard,
    records: '3.8M',
    features: '23',
    description: 'Monthly credit card snapshots',
    color: '#f97316',
    size: 'small'
  },
];

const getSizeClass = (size) => {
  switch (size) {
    case 'large': return 'md:col-span-2 md:row-span-2';
    case 'medium': return 'md:col-span-1 md:row-span-1';
    default: return 'md:col-span-1 md:row-span-1';
  }
};

export default function DataSources() {
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
            Data <span className="gradient-text">Sources</span>
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Seven interconnected data sources providing a comprehensive view of each applicant's credit history.
          </p>
        </motion.div>

        {/* Bento Grid */}
        <div className="grid md:grid-cols-4 gap-4 auto-rows-fr">
          {dataSources.map((source, index) => (
            <motion.div
              key={source.name}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={isInView ? { opacity: 1, scale: 1 } : {}}
              transition={{ duration: 0.5, delay: 0.1 + index * 0.1 }}
              whileHover={{ scale: 1.02, y: -5 }}
              className={`glass-card relative overflow-hidden group ${getSizeClass(source.size)}`}
            >
              {/* Glow effect */}
              <div 
                className="absolute inset-0 opacity-0 group-hover:opacity-10 transition-opacity duration-500"
                style={{ background: `radial-gradient(circle at center, ${source.color}, transparent)` }}
              />
              
              <div className="relative z-10 h-full flex flex-col">
                <div className="flex items-start justify-between mb-4">
                  <div 
                    className="w-12 h-12 rounded-xl flex items-center justify-center"
                    style={{ background: `${source.color}20` }}
                  >
                    <source.icon className="w-6 h-6" style={{ color: source.color }} />
                  </div>
                  <div 
                    className="px-2 py-1 rounded-md text-xs font-medium"
                    style={{ background: `${source.color}20`, color: source.color }}
                  >
                    {source.features} cols
                  </div>
                </div>
                
                <h3 className="font-semibold mb-2 text-sm md:text-base font-mono">{source.name}</h3>
                <p className="text-gray-400 text-sm flex-grow">{source.description}</p>
                
                <div className="mt-4 pt-4 border-t border-white/10">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">Records</span>
                    <span className="font-mono text-sm" style={{ color: source.color }}>{source.records}</span>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Summary Stats */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.8 }}
          className="mt-12 glass-card"
        >
          <div className="grid md:grid-cols-4 gap-6 text-center">
            {[
              { label: 'Total Records', value: '58M+', desc: 'Across all sources' },
              { label: 'Raw Features', value: '218', desc: 'Original columns' },
              { label: 'Engineered Features', value: '522', desc: 'After processing' },
              { label: 'Training Samples', value: '307K', desc: 'Labeled applications' },
            ].map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.4, delay: 0.9 + index * 0.1 }}
              >
                <div className="text-3xl font-bold gradient-text mb-1">{stat.value}</div>
                <div className="font-medium mb-1">{stat.label}</div>
                <div className="text-sm text-gray-500">{stat.desc}</div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
