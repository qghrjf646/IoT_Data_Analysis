"use client";

import { motion } from "framer-motion";
import { useState, useEffect } from "react";

interface GlitchTextProps {
  text: string;
  className?: string;
  delay?: number;
}

export default function GlitchText({ text, className = "", delay = 0 }: GlitchTextProps) {
  const [isGlitching, setIsGlitching] = useState(true);
  const [displayText, setDisplayText] = useState("");
  const glitchChars = "!@#$%^&*()_+-=[]{}|;':\",./<>?0123456789";

  useEffect(() => {
    const timeout = setTimeout(() => {
      let iteration = 0;
      const maxIterations = text.length * 3;

      const interval = setInterval(() => {
        setDisplayText(
          text
            .split("")
            .map((char, index) => {
              if (index < iteration / 3) {
                return char;
              }
              return glitchChars[Math.floor(Math.random() * glitchChars.length)];
            })
            .join("")
        );

        iteration++;

        if (iteration > maxIterations) {
          setDisplayText(text);
          setIsGlitching(false);
          clearInterval(interval);
        }
      }, 30);

      return () => clearInterval(interval);
    }, delay * 1000);

    return () => clearTimeout(timeout);
  }, [text, delay]);

  return (
    <motion.span
      className={`inline-block ${className}`}
      animate={isGlitching ? { x: [0, -2, 2, -1, 1, 0] } : {}}
      transition={{ duration: 0.1, repeat: isGlitching ? Infinity : 0 }}
    >
      <span className="relative">
        {displayText || text}
        {isGlitching && (
          <>
            <span
              className="absolute top-0 left-0 text-neon-primary opacity-70"
              style={{ clipPath: "inset(0 0 50% 0)", transform: "translate(-2px, 0)" }}
            >
              {displayText}
            </span>
            <span
              className="absolute top-0 left-0 text-neon-secondary opacity-70"
              style={{ clipPath: "inset(50% 0 0 0)", transform: "translate(2px, 0)" }}
            >
              {displayText}
            </span>
          </>
        )}
      </span>
    </motion.span>
  );
}
