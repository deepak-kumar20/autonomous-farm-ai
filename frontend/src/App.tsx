import React from 'react';
import { BrowserRouter, Routes, Route, useLocation, Navigate } from 'react-router-dom';
import { AnimatePresence, motion } from 'motion/react';
import { useFarm } from './hooks/useFarm';
import { FarmContext, useFarmContext } from './context/FarmContext';
import { TopBar, MobileNav } from './components/game/TopBar';
import {
  DayTransition,
  EventToasts,
  FlyerLayer,
  SeasonEndModal,
  LoadingScreen,
  OfflineScreen,
  ErrorToast,
} from './components/game/Overlays';
import { HomePage } from './pages/HomePage';
import { FarmPage } from './pages/FarmPage';
import { MarketPage } from './pages/MarketPage';
import { BarnPage } from './pages/BarnPage';
import { WeatherPage } from './pages/WeatherPage';
import { AIFarmerPage } from './pages/AIFarmerPage';
import { JournalPage } from './pages/JournalPage';

/** Pages that need a live farm session show a loader / offline state instead. */
const NeedsFarm: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const farm = useFarmContext();
  if (farm.farmState) return <>{children}</>;
  if (farm.isLoading) return <LoadingScreen />;
  return <OfflineScreen message={farm.error ?? 'Could not reach the farm server.'} onRetry={() => farm.initSession()} />;
};

const AnimatedRoutes: React.FC = () => {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait">
      <motion.main
        key={location.pathname}
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -12 }}
        transition={{ duration: 0.28, ease: 'easeOut' }}
        className="max-w-[1400px] mx-auto px-3 sm:px-5 pt-5 pb-28 lg:pb-12"
      >
        <Routes location={location}>
          <Route path="/" element={<HomePage />} />
          <Route path="/farm" element={<NeedsFarm><FarmPage /></NeedsFarm>} />
          <Route path="/market" element={<NeedsFarm><MarketPage /></NeedsFarm>} />
          <Route path="/barn" element={<NeedsFarm><BarnPage /></NeedsFarm>} />
          <Route path="/weather" element={<NeedsFarm><WeatherPage /></NeedsFarm>} />
          <Route path="/ai" element={<NeedsFarm><AIFarmerPage /></NeedsFarm>} />
          <Route path="/journal" element={<NeedsFarm><JournalPage /></NeedsFarm>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </motion.main>
    </AnimatePresence>
  );
};

export function App() {
  const farm = useFarm();
  return (
    <FarmContext.Provider value={farm}>
      <BrowserRouter>
        <TopBar />
        <AnimatedRoutes />
        <MobileNav />
        <DayTransition />
        <EventToasts />
        <FlyerLayer />
        <ErrorToast />
        <SeasonEndModal />
      </BrowserRouter>
    </FarmContext.Provider>
  );
}

export default App;
