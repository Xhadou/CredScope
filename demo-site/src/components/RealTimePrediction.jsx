import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';
import { 
  ArrowLeft, Send, Loader2, CheckCircle, AlertTriangle, XCircle, 
  User, Briefcase, CreditCard, Home, Car, TrendingUp, Info, FileText
} from 'lucide-react';

// API URL - change this to your FastAPI server address
const API_BASE_URL = 'http://localhost:8000';

const initialFormData = {
  // Required fields
  AMT_INCOME_TOTAL: 150000,
  AMT_CREDIT: 500000,
  DAYS_BIRTH: -12000, // ~33 years old
  
  // Credit Details
  AMT_ANNUITY: 25000,
  AMT_GOODS_PRICE: 450000,
  NAME_CONTRACT_TYPE: 'Cash loans',

  // Personal Info
  CODE_GENDER: 1, // 0=F, 1=M
  CNT_CHILDREN: 0,
  CNT_FAM_MEMBERS: 2,
  NAME_FAMILY_STATUS: 'Married',

  // Education
  NAME_EDUCATION_TYPE: 'Secondary / secondary special',

  // Housing & Assets
  NAME_HOUSING_TYPE: 'House / apartment',
  FLAG_OWN_CAR: 'N',
  FLAG_OWN_REALTY: 'Y',
  OWN_CAR_AGE: null,

  // Employment
  DAYS_EMPLOYED: -2000, // ~5.5 years
  NAME_INCOME_TYPE: 'Working',
  OCCUPATION_TYPE: 'Laborers',
  ORGANIZATION_TYPE: 'Business Entity Type 3',

  // Contact
  FLAG_MOBIL: 1,
  FLAG_PHONE: 0,
  FLAG_EMAIL: 0,
  FLAG_WORK_PHONE: 0,

  // External scores (most important features) - 0 to 1
  EXT_SOURCE_1: 0.5,
  EXT_SOURCE_2: 0.5,
  EXT_SOURCE_3: 0.5,

  // Region
  REGION_POPULATION_RELATIVE: 0.02,
  REGION_RATING_CLIENT: 2,

  // Documents
  FLAG_DOCUMENT_3: 1,
  FLAG_DOCUMENT_6: 0,
  FLAG_DOCUMENT_8: 0,
};

const genderOptions = [{ value: 0, label: 'Female' }, { value: 1, label: 'Male' }];
const incomeTypes = ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student', 'Unemployed'];
const educationTypes = ['Lower secondary', 'Secondary / secondary special', 'Incomplete higher', 'Higher education', 'Academic degree'];
const housingTypes = ['House / apartment', 'Rented apartment', 'With parents', 'Municipal apartment', 'Office apartment', 'Co-op apartment'];
const contractTypes = ['Cash loans', 'Revolving loans'];
const familyStatuses = ['Single / not married', 'Married', 'Civil marriage', 'Widow', 'Separated'];
const occupationTypes = ['Laborers', 'Core staff', 'Accountants', 'Managers', 'Drivers', 'Sales staff', 'Cleaning staff', 'Cooking staff', 'Private service staff', 'Medicine staff', 'Security staff', 'High skill tech staff', 'Waiters/barmen staff', 'Low-skill Laborers', 'Realty agents', 'Secretaries', 'IT staff', 'HR staff'];

const getDecisionStyle = (decision) => {
  const d = decision?.toUpperCase();
  switch (d) {
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

export default function RealTimePrediction({ onBack }) {
  const [formData, setFormData] = useState(initialFormData);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('personal');

  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const calculateAge = (daysBirth) => {
    return Math.floor(Math.abs(daysBirth) / 365);
  };

  const calculateEmploymentYears = (daysEmployed) => {
    if (!daysEmployed || daysEmployed > 0) return 0;
    return Math.floor(Math.abs(daysEmployed) / 365);
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    // Build the API request payload with all form fields
    const payload = { ...formData };
    
    // Remove null values
    Object.keys(payload).forEach(key => {
      if (payload[key] === null || payload[key] === '') {
        delete payload[key];
      }
    });

    try {
      // Try the explain endpoint first for more detailed results
      const response = await fetch(`${API_BASE_URL}/predict/explain`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `API Error: ${response.status}`);
      }

      const data = await response.json();
      
      // Map API response to our format
      setResult({
        probability: data.default_probability,
        decision: data.decision,
        confidence: data.confidence,
        risk_level: data.risk_level,
        top_features: data.top_features || [],
        timestamp: data.timestamp,
        is_api: true
      });
      setError(null);

    } catch (err) {
      console.error('API Error:', err);
      
      // Try simple predict endpoint as fallback
      try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
        });

        if (response.ok) {
          const data = await response.json();
          setResult({
            probability: data.default_probability,
            decision: data.decision,
            confidence: data.confidence,
            risk_level: data.risk_level,
            top_features: [],
            timestamp: data.timestamp,
            is_api: true
          });
          setError(null);
          setIsLoading(false);
          return;
        }
      } catch (fallbackErr) {
        console.error('Fallback API Error:', fallbackErr);
      }

      // If all API calls fail, show demo simulation
      setError(`API unavailable: ${err.message}. Showing simulated prediction.`);
      
      const avgExtScore = (formData.EXT_SOURCE_1 + formData.EXT_SOURCE_2 + formData.EXT_SOURCE_3) / 3;
      const dtiRatio = formData.AMT_CREDIT / formData.AMT_INCOME_TOTAL;
      const employmentYears = calculateEmploymentYears(formData.DAYS_EMPLOYED);
      
      // Simple heuristic for demo
      let riskScore = 0.5;
      riskScore -= (avgExtScore - 0.5) * 0.6;
      riskScore += (dtiRatio - 3) * 0.05;
      riskScore -= (employmentYears - 3) * 0.02;
      // Adjust for assets
      if (formData.FLAG_OWN_REALTY === 'Y') riskScore -= 0.03;
      if (formData.FLAG_OWN_CAR === 'Y') riskScore -= 0.02;
      riskScore = Math.max(0.05, Math.min(0.95, riskScore));
      
      let decision = 'REVIEW';
      if (riskScore < 0.2) decision = 'APPROVE';
      else if (riskScore > 0.5) decision = 'REJECT';

      setResult({
        probability: riskScore,
        decision: decision,
        confidence: 1 - Math.abs(riskScore - 0.5) * 2,
        risk_level: riskScore < 0.2 ? 'LOW' : riskScore < 0.5 ? 'MEDIUM' : 'HIGH',
        top_features: [
          { feature: 'EXT_SOURCE_2', importance: 0.15, value: formData.EXT_SOURCE_2 },
          { feature: 'EXT_SOURCE_3', importance: 0.12, value: formData.EXT_SOURCE_3 },
          { feature: 'EXT_SOURCE_1', importance: 0.10, value: formData.EXT_SOURCE_1 },
          { feature: 'DAYS_BIRTH', importance: 0.08, value: formData.DAYS_BIRTH },
          { feature: 'AMT_CREDIT', importance: 0.06, value: formData.AMT_CREDIT },
        ],
        is_demo: true
      });
    } finally {
      setIsLoading(false);
    }
  };

  const decisionStyle = result ? getDecisionStyle(result.decision) : null;
  const DecisionIcon = decisionStyle?.icon;

  const tabs = [
    { id: 'personal', label: 'Personal', icon: User },
    { id: 'employment', label: 'Employment', icon: Briefcase },
    { id: 'credit', label: 'Credit Details', icon: CreditCard },
    { id: 'assets', label: 'Assets & Housing', icon: Home },
    { id: 'scores', label: 'Credit Scores', icon: TrendingUp },
  ];

  return (
    <div className="min-h-screen bg-gray-950 pt-20">
      <div className="max-w-6xl mx-auto px-6 py-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <button
            onClick={onBack}
            className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors mb-6"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Back to Home</span>
          </button>
          
          <h1 className="text-4xl font-bold mb-4">
            Real-Time <span className="gradient-text">Prediction</span>
          </h1>
          <p className="text-gray-400 text-lg">
            Enter applicant details to get an instant credit risk assessment using our ensemble ML models.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Form Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-2"
          >
            <div className="glass-card">
              {/* Tabs */}
              <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg whitespace-nowrap transition-all ${
                      activeTab === tab.id 
                        ? 'bg-indigo-500 text-white' 
                        : 'bg-white/5 text-gray-400 hover:bg-white/10'
                    }`}
                  >
                    <tab.icon className="w-4 h-4" />
                    <span>{tab.label}</span>
                  </button>
                ))}
              </div>

              {/* Personal Info Tab */}
              {activeTab === 'personal' && (
                <div className="space-y-6">
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Gender *</label>
                      <select
                        value={formData.CODE_GENDER}
                        onChange={(e) => handleInputChange('CODE_GENDER', parseInt(e.target.value))}
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
                      >
                        {genderOptions.map((opt) => (
                          <option key={opt.value} value={opt.value} className="bg-gray-900">{opt.label}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Age (years) *</label>
                      <input
                        type="number"
                        value={calculateAge(formData.DAYS_BIRTH)}
                        onChange={(e) => handleInputChange('DAYS_BIRTH', -parseInt(e.target.value || 30) * 365)}
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
                        min="18"
                        max="80"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Family Status</label>
                      <select
                        value={formData.NAME_FAMILY_STATUS}
                        onChange={(e) => handleInputChange('NAME_FAMILY_STATUS', e.target.value)}
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
                      >
                        {familyStatuses.map((status) => (
                          <option key={status} value={status} className="bg-gray-900">{status}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Number of Children</label>
                      <input
                        type="number"
                        value={formData.CNT_CHILDREN}
                        onChange={(e) => handleInputChange('CNT_CHILDREN', parseInt(e.target.value || 0))}
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
                        min="0"
                        max="20"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Family Members</label>
                      <input
                        type="number"
                        value={formData.CNT_FAM_MEMBERS}
                        onChange={(e) => handleInputChange('CNT_FAM_MEMBERS', parseFloat(e.target.value || 1))}
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
                        min="1"
                        max="20"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Education Level</label>
                      <select
                        value={formData.NAME_EDUCATION_TYPE}
                        onChange={(e) => handleInputChange('NAME_EDUCATION_TYPE', e.target.value)}
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
                      >
                        {educationTypes.map((type) => (
                          <option key={type} value={type} className="bg-gray-900">{type}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                  
                  {/* Contact Info */}
                  <div className="pt-4 border-t border-white/10">
                    <h4 className="text-sm font-medium text-gray-300 mb-4">Contact Information</h4>
                    <div className="flex flex-wrap gap-6">
                      {[
                        { key: 'FLAG_MOBIL', label: 'Mobile Phone' },
                        { key: 'FLAG_PHONE', label: 'Home Phone' },
                        { key: 'FLAG_WORK_PHONE', label: 'Work Phone' },
                        { key: 'FLAG_EMAIL', label: 'Email' },
                      ].map((item) => (
                        <label key={item.key} className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={formData[item.key] === 1}
                            onChange={(e) => handleInputChange(item.key, e.target.checked ? 1 : 0)}
                            className="w-5 h-5 rounded border-white/20 bg-white/5 text-indigo-500"
                          />
                          <span className="text-gray-400">{item.label}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Employment Tab */}
              {activeTab === 'employment' && (
                <div className="space-y-6">
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Income Type</label>
                      <select
                        value={formData.NAME_INCOME_TYPE}
                        onChange={(e) => handleInputChange('NAME_INCOME_TYPE', e.target.value)}
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
                      >
                        {incomeTypes.map((type) => (
                          <option key={type} value={type} className="bg-gray-900">{type}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Annual Income ($) *</label>
                      <input
                        type="number"
                        value={formData.AMT_INCOME_TOTAL}
                        onChange={(e) => handleInputChange('AMT_INCOME_TOTAL', parseFloat(e.target.value || 0))}
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
                        min="0"
                        step="1000"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Years Employed</label>
                      <input
                        type="number"
                        value={calculateEmploymentYears(formData.DAYS_EMPLOYED)}
                        onChange={(e) => handleInputChange('DAYS_EMPLOYED', -parseInt(e.target.value || 0) * 365)}
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
                        min="0"
                        max="50"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Occupation</label>
                      <select
                        value={formData.OCCUPATION_TYPE || ''}
                        onChange={(e) => handleInputChange('OCCUPATION_TYPE', e.target.value)}
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
                      >
                        <option value="" className="bg-gray-900">Select occupation...</option>
                        {occupationTypes.map((type) => (
                          <option key={type} value={type} className="bg-gray-900">{type}</option>
                        ))}
                      </select>
                    </div>
                    <div className="md:col-span-2">
                      <label className="block text-sm text-gray-400 mb-2">Organization Type</label>
                      <input
                        type="text"
                        value={formData.ORGANIZATION_TYPE || ''}
                        onChange={(e) => handleInputChange('ORGANIZATION_TYPE', e.target.value)}
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
                        placeholder="e.g., Business Entity Type 3, Self-employed, Government"
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Credit Details Tab */}
              {activeTab === 'credit' && (
                <div className="space-y-6">
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Contract Type</label>
                      <select
                        value={formData.NAME_CONTRACT_TYPE}
                        onChange={(e) => handleInputChange('NAME_CONTRACT_TYPE', e.target.value)}
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
                      >
                        {contractTypes.map((type) => (
                          <option key={type} value={type} className="bg-gray-900">{type}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Credit Amount ($) *</label>
                      <input
                        type="number"
                        value={formData.AMT_CREDIT}
                        onChange={(e) => handleInputChange('AMT_CREDIT', parseFloat(e.target.value || 0))}
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
                        min="0"
                        step="10000"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Annuity Payment ($)</label>
                      <input
                        type="number"
                        value={formData.AMT_ANNUITY}
                        onChange={(e) => handleInputChange('AMT_ANNUITY', parseFloat(e.target.value || 0))}
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
                        min="0"
                        step="1000"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Goods Price ($)</label>
                      <input
                        type="number"
                        value={formData.AMT_GOODS_PRICE}
                        onChange={(e) => handleInputChange('AMT_GOODS_PRICE', parseFloat(e.target.value || 0))}
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
                        min="0"
                        step="10000"
                      />
                    </div>
                  </div>
                  
                  {/* DTI Indicator */}
                  <div className="bg-white/5 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-gray-400">Debt-to-Income Ratio</span>
                      <span className={`font-medium ${
                        (formData.AMT_CREDIT / formData.AMT_INCOME_TOTAL) <= 3 ? 'text-green-400' :
                        (formData.AMT_CREDIT / formData.AMT_INCOME_TOTAL) <= 5 ? 'text-amber-400' : 'text-red-400'
                      }`}>
                        {(formData.AMT_CREDIT / formData.AMT_INCOME_TOTAL).toFixed(2)}x
                      </span>
                    </div>
                    <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${
                          (formData.AMT_CREDIT / formData.AMT_INCOME_TOTAL) <= 3 ? 'bg-green-500' :
                          (formData.AMT_CREDIT / formData.AMT_INCOME_TOTAL) <= 5 ? 'bg-amber-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${Math.min((formData.AMT_CREDIT / formData.AMT_INCOME_TOTAL) / 10 * 100, 100)}%` }}
                      />
                    </div>
                    <p className="text-xs text-gray-500 mt-2">Recommended: Below 3x for best approval chances</p>
                  </div>
                </div>
              )}

              {/* Assets & Housing Tab */}
              {activeTab === 'assets' && (
                <div className="space-y-6">
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Housing Type</label>
                      <select
                        value={formData.NAME_HOUSING_TYPE}
                        onChange={(e) => handleInputChange('NAME_HOUSING_TYPE', e.target.value)}
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
                      >
                        {housingTypes.map((type) => (
                          <option key={type} value={type} className="bg-gray-900">{type}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Region Rating</label>
                      <select
                        value={formData.REGION_RATING_CLIENT || 2}
                        onChange={(e) => handleInputChange('REGION_RATING_CLIENT', parseInt(e.target.value))}
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
                      >
                        <option value={1} className="bg-gray-900">1 - Best</option>
                        <option value={2} className="bg-gray-900">2 - Average</option>
                        <option value={3} className="bg-gray-900">3 - Below Average</option>
                      </select>
                    </div>
                  </div>
                  
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="bg-white/5 rounded-lg p-4">
                      <label className="flex items-center gap-3 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={formData.FLAG_OWN_REALTY === 'Y'}
                          onChange={(e) => handleInputChange('FLAG_OWN_REALTY', e.target.checked ? 'Y' : 'N')}
                          className="w-6 h-6 rounded border-white/20 bg-white/5 text-indigo-500"
                        />
                        <div>
                          <Home className="w-6 h-6 text-indigo-400 mb-1" />
                          <span className="font-medium">Owns Real Estate</span>
                          <p className="text-xs text-gray-500">Property ownership</p>
                        </div>
                      </label>
                    </div>
                    <div className="bg-white/5 rounded-lg p-4">
                      <label className="flex items-center gap-3 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={formData.FLAG_OWN_CAR === 'Y'}
                          onChange={(e) => handleInputChange('FLAG_OWN_CAR', e.target.checked ? 'Y' : 'N')}
                          className="w-6 h-6 rounded border-white/20 bg-white/5 text-indigo-500"
                        />
                        <div>
                          <Car className="w-6 h-6 text-cyan-400 mb-1" />
                          <span className="font-medium">Owns a Car</span>
                          <p className="text-xs text-gray-500">Vehicle ownership</p>
                        </div>
                      </label>
                    </div>
                  </div>

                  {formData.FLAG_OWN_CAR === 'Y' && (
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Car Age (years)</label>
                      <input
                        type="number"
                        value={formData.OWN_CAR_AGE || ''}
                        onChange={(e) => handleInputChange('OWN_CAR_AGE', e.target.value ? parseFloat(e.target.value) : null)}
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
                        min="0"
                        max="50"
                        placeholder="Age of vehicle"
                      />
                    </div>
                  )}

                  {/* Documents */}
                  <div className="pt-4 border-t border-white/10">
                    <h4 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-2">
                      <FileText className="w-4 h-4" />
                      Documents Provided
                    </h4>
                    <div className="flex flex-wrap gap-4">
                      {[
                        { key: 'FLAG_DOCUMENT_3', label: 'Government ID' },
                        { key: 'FLAG_DOCUMENT_6', label: 'Proof of Income' },
                        { key: 'FLAG_DOCUMENT_8', label: 'Proof of Address' },
                      ].map((doc) => (
                        <label key={doc.key} className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={formData[doc.key] === 1}
                            onChange={(e) => handleInputChange(doc.key, e.target.checked ? 1 : 0)}
                            className="w-5 h-5 rounded border-white/20 bg-white/5 text-indigo-500"
                          />
                          <span className="text-gray-400">{doc.label}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Credit Scores Tab */}
              {activeTab === 'scores' && (
                <div className="space-y-6">
                  <div className="bg-indigo-500/10 border border-indigo-500/20 rounded-lg p-4 flex items-start gap-3">
                    <Info className="w-5 h-5 text-indigo-400 mt-0.5" />
                    <div>
                      <p className="text-sm text-indigo-200">These are the most important features for prediction.</p>
                      <p className="text-xs text-indigo-300/70 mt-1">External credit scores from third-party bureaus significantly impact the risk assessment. Higher scores = lower risk.</p>
                    </div>
                  </div>
                  
                  {[
                    { key: 'EXT_SOURCE_2', label: 'External Source 2', desc: 'Most Important - Primary credit bureau score' },
                    { key: 'EXT_SOURCE_3', label: 'External Source 3', desc: 'Secondary credit bureau score' },
                    { key: 'EXT_SOURCE_1', label: 'External Source 1', desc: 'Tertiary credit bureau score' },
                  ].map((source) => (
                    <div key={source.key}>
                      <div className="flex items-center justify-between mb-2">
                        <div>
                          <label className="text-sm text-gray-300">{source.label}</label>
                          <p className="text-xs text-gray-500">{source.desc}</p>
                        </div>
                        <span className={`font-medium text-lg ${
                          formData[source.key] >= 0.6 ? 'text-green-400' :
                          formData[source.key] >= 0.4 ? 'text-amber-400' : 'text-red-400'
                        }`}>
                          {(formData[source.key] * 100).toFixed(0)}%
                        </span>
                      </div>
                      <input
                        type="range"
                        value={formData[source.key]}
                        onChange={(e) => handleInputChange(source.key, parseFloat(e.target.value))}
                        className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                        min="0"
                        max="1"
                        step="0.01"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>0% (Poor)</span>
                        <span>50%</span>
                        <span>100% (Excellent)</span>
                      </div>
                    </div>
                  ))}

                  {/* Average Score Summary */}
                  <div className="bg-white/5 rounded-lg p-4 mt-4">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Average External Score</span>
                      <span className={`text-2xl font-bold ${
                        ((formData.EXT_SOURCE_1 + formData.EXT_SOURCE_2 + formData.EXT_SOURCE_3) / 3) >= 0.6 ? 'text-green-400' :
                        ((formData.EXT_SOURCE_1 + formData.EXT_SOURCE_2 + formData.EXT_SOURCE_3) / 3) >= 0.4 ? 'text-amber-400' : 'text-red-400'
                      }`}>
                        {(((formData.EXT_SOURCE_1 + formData.EXT_SOURCE_2 + formData.EXT_SOURCE_3) / 3) * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Submit Button */}
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleSubmit}
                disabled={isLoading}
                className="w-full mt-8 px-6 py-4 bg-gradient-to-r from-indigo-500 to-cyan-500 rounded-xl font-semibold text-lg flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Analyzing with Ensemble Models...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-5 h-5" />
                    <span>Get Prediction</span>
                  </>
                )}
              </motion.button>
            </div>
          </motion.div>

          {/* Results Section */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-1"
          >
            <div className="glass-card sticky top-24">
              <h3 className="text-xl font-semibold mb-6">Prediction Result</h3>
              
              {!result && !isLoading && (
                <div className="text-center py-12 text-gray-500">
                  <TrendingUp className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Fill in the form and click "Get Prediction" to see results</p>
                </div>
              )}

              {isLoading && (
                <div className="text-center py-12">
                  <Loader2 className="w-12 h-12 mx-auto mb-4 animate-spin text-indigo-500" />
                  <p className="text-gray-400">Processing with ensemble models...</p>
                  <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
                    {['LightGBM', 'XGBoost', 'CatBoost'].map((model, i) => (
                      <div key={model} className="text-center">
                        <div className="w-2 h-2 rounded-full mx-auto mb-1 bg-indigo-500 animate-pulse" style={{ animationDelay: `${i * 0.2}s` }} />
                        <span className="text-gray-500">{model}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <AnimatePresence>
                {result && !isLoading && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="space-y-6"
                  >
                    {error && (
                      <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-3 text-sm text-amber-300">
                        {error}
                      </div>
                    )}

                    {/* Risk Score */}
                    <div className="text-center">
                      <div className="text-sm text-gray-400 mb-2">Risk Score</div>
                      <div className="text-5xl font-bold gradient-text">
                        {(result.probability * 100).toFixed(1)}%
                      </div>
                    </div>

                    {/* Gauge */}
                    <div className="relative h-3 rounded-full overflow-hidden">
                      <div className="absolute inset-0 flex">
                        <div className="w-1/5 bg-green-500/30" />
                        <div className="bg-amber-500/30" style={{ width: '30%' }} />
                        <div className="flex-1 bg-red-500/30" />
                      </div>
                      <motion.div
                        initial={{ left: 0 }}
                        animate={{ left: `${result.probability * 100}%` }}
                        className="absolute top-0 w-1 h-full bg-white rounded-full shadow-lg"
                        style={{ transform: 'translateX(-50%)' }}
                      />
                    </div>
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>0%</span>
                      <span>20%</span>
                      <span>50%</span>
                      <span>100%</span>
                    </div>

                    {/* Decision */}
                    <div className={`p-4 rounded-xl border ${decisionStyle.bg} ${decisionStyle.border}`}>
                      <div className="flex items-center gap-3">
                        <DecisionIcon className={`w-8 h-8 ${decisionStyle.text}`} />
                        <div>
                          <div className={`text-xl font-bold ${decisionStyle.text}`}>
                            {result.decision}
                          </div>
                          <div className="text-sm text-gray-400">
                            {result.decision === 'APPROVE' && 'Low risk - Auto approve'}
                            {result.decision === 'REVIEW' && 'Medium risk - Manual review'}
                            {result.decision === 'REJECT' && 'High risk - Auto reject'}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Model Scores */}
                    {result.model_scores && (
                      <div>
                        <div className="text-sm text-gray-400 mb-3">Individual Model Scores</div>
                        <div className="space-y-2">
                          {Object.entries(result.model_scores).map(([model, score]) => (
                            <div key={model} className="flex items-center justify-between text-sm">
                              <span className="text-gray-400 capitalize">{model}</span>
                              <span className="font-mono">{(score * 100).toFixed(2)}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Top Features */}
                    {result.top_features && (
                      <div>
                        <div className="text-sm text-gray-400 mb-3">Top Contributing Features</div>
                        <div className="space-y-2">
                          {result.top_features.slice(0, 5).map((feat, i) => (
                            <div key={i} className="text-sm">
                              <div className="flex justify-between mb-1">
                                <span className="text-gray-400 font-mono text-xs">{feat.feature}</span>
                                <span className="text-xs">{feat.importance.toFixed(1)}%</span>
                              </div>
                              <div className="h-1 rounded-full bg-white/10 overflow-hidden">
                                <div 
                                  className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-cyan-500"
                                  style={{ width: `${Math.min(feat.importance * 2, 100)}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
