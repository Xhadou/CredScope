import { motion } from 'framer-motion';
import { useInView } from 'framer-motion';
import { useRef } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell } from 'recharts';

const topFeatures = [
  { name: 'EXT_SOURCE_2', importance: 100, category: 'External' },
  { name: 'EXT_SOURCE_3', importance: 89, category: 'External' },
  { name: 'EXT_SOURCE_1', importance: 72, category: 'External' },
  { name: 'DAYS_BIRTH', importance: 58, category: 'Application' },
  { name: 'AMT_CREDIT', importance: 52, category: 'Application' },
  { name: 'AMT_ANNUITY', importance: 48, category: 'Application' },
  { name: 'DAYS_EMPLOYED', importance: 45, category: 'Application' },
  { name: 'AMT_GOODS_PRICE', importance: 42, category: 'Application' },
  { name: 'BUREAU_ACTIVE_COUNT', importance: 38, category: 'Bureau' },
  { name: 'PREV_APP_APPROVED_COUNT', importance: 35, category: 'Previous' },
];

const categories = [
  { name: 'External', count: 3, color: '#22c55e', description: 'External credit scores' },
  { name: 'Application', count: 95, color: '#6366f1', description: 'Current application data' },
  { name: 'Bureau', count: 156, color: '#f59e0b', description: 'Credit bureau history' },
  { name: 'Previous', count: 89, color: '#ec4899', description: 'Previous applications' },
  { name: 'Installments', count: 72, color: '#14b8a6', description: 'Payment behavior' },
  { name: 'POS/Cash', count: 68, color: '#8b5cf6', description: 'POS and cash loans' },
  { name: 'Credit Card', count: 39, color: '#f97316', description: 'Card balance data' },
];

const getCategoryColor = (category) => {
  const cat = categories.find(c => c.name === category);
  return cat ? cat.color : '#6366f1';
};

export default function FeatureImportance() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  return (
    <section className="py-32 relative" ref={ref}>
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-indigo-500/5 to-transparent" />
      
      <div className="max-w-7xl mx-auto px-6 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center mb-20"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            Feature <span className="gradient-text">Importance</span>
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            522 features engineered from 7 data sources. External credit scores are the strongest predictors.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Top Features Chart */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="glass-card"
          >
            <h3 className="text-xl font-semibold mb-6">Top 10 Features</h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={topFeatures} layout="vertical" margin={{ left: 20 }}>
                <XAxis type="number" domain={[0, 100]} stroke="#9ca3af" />
                <YAxis 
                  dataKey="name" 
                  type="category" 
                  stroke="#9ca3af" 
                  width={150}
                  tick={{ fontSize: 12 }}
                />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                  {topFeatures.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={getCategoryColor(entry.category)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Category Distribution */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="glass-card"
          >
            <h3 className="text-xl font-semibold mb-6">Feature Categories</h3>
            <div className="space-y-4">
              {categories.map((category, index) => (
                <motion.div
                  key={category.name}
                  initial={{ opacity: 0, x: 20 }}
                  animate={isInView ? { opacity: 1, x: 0 } : {}}
                  transition={{ duration: 0.4, delay: 0.4 + index * 0.1 }}
                  className="group"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <div 
                        className="w-3 h-3 rounded-full"
                        style={{ background: category.color }}
                      />
                      <span className="font-medium">{category.name}</span>
                    </div>
                    <span className="text-gray-400">{category.count} features</span>
                  </div>
                  <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={isInView ? { width: `${(category.count / 156) * 100}%` } : {}}
                      transition={{ duration: 0.8, delay: 0.5 + index * 0.1 }}
                      className="h-full rounded-full"
                      style={{ background: category.color }}
                    />
                  </div>
                  <p className="text-sm text-gray-500 mt-1">{category.description}</p>
                </motion.div>
              ))}
            </div>
            
            {/* Total count */}
            <div className="mt-8 pt-6 border-t border-white/10 flex items-center justify-between">
              <span className="text-gray-400">Total Features</span>
              <span className="text-2xl font-bold gradient-text">522</span>
            </div>
          </motion.div>
        </div>

        {/* Feature engineering highlights */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="mt-12 grid md:grid-cols-3 gap-6"
        >
          {[
            { title: 'Aggregations', desc: 'Min, max, mean, sum, count for each data source', count: '150+' },
            { title: 'Time Features', desc: 'Days since, months active, payment timing', count: '80+' },
            { title: 'Cross Features', desc: 'Ratios, interactions between key variables', count: '60+' },
          ].map((item, index) => (
            <motion.div
              key={item.title}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={isInView ? { opacity: 1, scale: 1 } : {}}
              transition={{ duration: 0.4, delay: 0.7 + index * 0.1 }}
              whileHover={{ scale: 1.03 }}
              className="glass-card text-center"
            >
              <div className="text-3xl font-bold gradient-text mb-2">{item.count}</div>
              <h4 className="font-semibold mb-2">{item.title}</h4>
              <p className="text-sm text-gray-400">{item.desc}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
