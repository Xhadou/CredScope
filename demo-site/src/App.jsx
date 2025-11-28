import { useState } from 'react'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import ProblemSolution from './components/ProblemSolution'
import PerformanceCharts from './components/PerformanceCharts'
import EnsembleArchitecture from './components/EnsembleArchitecture'
import FeatureImportance from './components/FeatureImportance'
import DataSources from './components/DataSources'
import LiveDemo from './components/LiveDemo'
import DecisionThresholds from './components/DecisionThresholds'
import Footer from './components/Footer'
import RealTimePrediction from './components/RealTimePrediction'

function App() {
  const [currentPage, setCurrentPage] = useState('home')

  if (currentPage === 'predict') {
    return (
      <div className="min-h-screen bg-gray-950">
        <Navbar onNavigate={setCurrentPage} currentPage={currentPage} />
        <RealTimePrediction onBack={() => setCurrentPage('home')} />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-950">
      <Navbar onNavigate={setCurrentPage} currentPage={currentPage} />
      <main>
        <Hero />
        <ProblemSolution />
        <PerformanceCharts />
        <EnsembleArchitecture />
        <FeatureImportance />
        <DataSources />
        <LiveDemo />
        <DecisionThresholds />
      </main>
      <Footer />
    </div>
  )
}

export default App
