"use client";

import { useEffect, useCallback, ReactNode, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { TOTAL_SLIDES, SLIDES } from "../lib/slides-config";
import ParticleBackground from "./ParticleBackground";

interface SlideContainerProps {
  children: ReactNode;
  slideNumber: number;
}

export default function SlideContainer({ children, slideNumber }: SlideContainerProps) {
  const router = useRouter();
  const [scale, setScale] = useState(1);
  const [isWarping, setIsWarping] = useState(false);
  const currentSlide = SLIDES.find(s => s.id === slideNumber);

  const handleResize = useCallback(() => {
    const targetWidth = 1920;
    const targetHeight = 1080;
    const widthScale = window.innerWidth / targetWidth;
    const heightScale = window.innerHeight / targetHeight;
    setScale(Math.min(widthScale, heightScale));
  }, []);

  const navigate = useCallback(
    (direction: "prev" | "next") => {
      if (direction === "next" && slideNumber < TOTAL_SLIDES) {
        setIsWarping(true);
        setTimeout(() => {
          router.push(`/slides/${slideNumber + 1}`);
          setTimeout(() => setIsWarping(false), 300);
        }, 50);
      } else if (direction === "prev" && slideNumber > 1) {
        setIsWarping(true);
        setTimeout(() => {
          router.push(`/slides/${slideNumber - 1}`);
          setTimeout(() => setIsWarping(false), 300);
        }, 50);
      }
    },
    [router, slideNumber]
  );

  useEffect(() => {
    handleResize();
    window.addEventListener("resize", handleResize);
    
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight" || e.key === " ") {
        e.preventDefault();
        navigate("next");
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        navigate("prev");
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [navigate, handleResize]);

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-neon-bg flex items-center justify-center" style={{ perspective: "2000px" }}>
      <div className="scanline" />
      
      <ParticleBackground 
        intensity={currentSlide?.bgIntensity || 100} 
        color={currentSlide?.bgColor || "#00f0ff"} 
        warpSpeed={isWarping}
      />

      <AnimatePresence>
        {isWarping && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.2 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-100 bg-white pointer-events-none"
          />
        )}
      </AnimatePresence>

      <div 
        style={{ 
          transform: `scale(${scale})`,
          transformOrigin: "top center",
          width: "1920px",
          height: "1080px",
          flexShrink: 0,
          marginTop: `${Math.max(0, (window.innerHeight - 1080 * scale) / 2)}px`
        }}
        className={`relative z-10 overflow-hidden flex flex-col ${isWarping ? "chromatic-aberration" : ""}`}
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={slideNumber}
            initial={{ 
              opacity: 0, 
              scale: 0.8,
              rotateY: 20,
              z: -500,
              filter: "blur(10px)"
            }}
            animate={{ 
              opacity: 1, 
              scale: 1,
              rotateY: 0,
              z: 0,
              filter: "blur(0px)"
            }}
            exit={{ 
              opacity: 0, 
              scale: 1.2,
              rotateY: -20,
              z: 200,
              filter: "blur(10px)"
            }}
            transition={{ 
              duration: 0.6, 
              ease: [0.22, 1, 0.36, 1] 
            }}
            className="w-full h-full flex flex-col p-8"
          >
            {children}
          </motion.div>
        </AnimatePresence>
      </div>

      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-4 z-50">
        <div className="flex gap-2">
          {Array.from({ length: TOTAL_SLIDES }, (_, i) => (
            <motion.div
              key={i}
              className={`w-2 h-2 rounded-full transition-all duration-300 ${
                i + 1 === slideNumber
                  ? "bg-neon-primary shadow-neon-cyan scale-125"
                  : i + 1 < slideNumber
                  ? "bg-neon-primary/50"
                  : "bg-neon-muted/30"
              }`}
              whileHover={{ scale: 1.5 }}
            />
          ))}
        </div>
        <span className="text-neon-muted text-sm font-mono">
          {slideNumber} / {TOTAL_SLIDES}
        </span>
      </div>

      <div className="absolute bottom-6 right-6 text-neon-muted/50 text-xs hidden sm:block">
        ← → pour naviguer
      </div>
    </div>
  );
}
