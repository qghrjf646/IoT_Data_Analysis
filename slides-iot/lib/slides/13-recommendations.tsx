"use client";

import { motion } from "framer-motion";
import NeonCard from "@/components/NeonCard";
import { Layers, Shield, RefreshCw } from "lucide-react";

export default function Slide13Recommendations() {
  const layers = [
    {
      num: 1,
      name: "LOF",
      desc: "Zero-day attack early warning",
      color: "cyan",
    },
    {
      num: 2,
      name: "Random Forest",
      desc: "Classification with best accuracy and adversarial robustness",
      color: "violet",
    },
    {
      num: 3,
      name: "Defense",
      desc: "Input validation and adversarial training",
      color: "magenta",
    },
  ];

  const hardening = [
    "Implement adversarial training with augmented samples",
    "Regular model retraining with new threat intelligence",
    "Feature monitoring for distribution drift",
  ];

  return (
    <div className="relative flex-1 flex flex-col p-8 overflow-hidden">

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-5xl md:text-7xl font-display font-bold text-neon-accent mb-2 leading-tight"
      >
        Recommendations
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="relative z-10 text-xl text-neon-muted mb-8"
      >
        Multi-Layer Defense Architecture
      </motion.p>

      <div className="relative z-10 flex-1 grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="space-y-4">
          {layers.map((layer, i) => (
            <motion.div
              key={layer.num}
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 + i * 0.2 }}
              className={`
                p-6 rounded-xl border-2 flex items-center gap-6
                ${layer.color === "cyan" ? "bg-neon-primary/10 border-neon-primary/30" : ""}
                ${layer.color === "violet" ? "bg-neon-accent/10 border-neon-accent/30" : ""}
                ${layer.color === "magenta" ? "bg-neon-secondary/10 border-neon-secondary/30" : ""}
              `}
            >
              <div className={`
                w-16 h-16 rounded-xl flex items-center justify-center text-3xl font-display font-bold
                ${layer.color === "cyan" ? "bg-neon-primary/20 text-neon-primary" : ""}
                ${layer.color === "violet" ? "bg-neon-accent/20 text-neon-accent" : ""}
                ${layer.color === "magenta" ? "bg-neon-secondary/20 text-neon-secondary" : ""}
              `}>
                L{layer.num}
              </div>
              <div>
                <h4 className={`
                  text-xl font-bold mb-1
                  ${layer.color === "cyan" ? "text-neon-primary" : ""}
                  ${layer.color === "violet" ? "text-neon-accent" : ""}
                  ${layer.color === "magenta" ? "text-neon-secondary" : ""}
                `}>
                  {layer.name}
                </h4>
                <p className="text-neon-muted">{layer.desc}</p>
              </div>
            </motion.div>
          ))}
        </div>

        <NeonCard delay={0.8} glow="violet" title="Production Hardening">
          <div className="mt-6 space-y-4">
            {hardening.map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 1.0 + i * 0.15 }}
                className="flex items-start gap-4"
              >
                <div className="w-2 h-2 rounded-full bg-neon-accent mt-2 shrink-0" />
                <span className="text-neon-text">{item}</span>
              </motion.div>
            ))}
          </div>
        </NeonCard>
      </div>
    </div>
  );
}
