"use client";

import { motion } from "framer-motion";
import NeonCard from "@/components/NeonCard";

export default function Slide10CausativeAttack() {
  const poisonResults = [
    { rate: "0%", acc: "68.8%", desc: "Baseline" },
    { rate: "10%", acc: "67.0%", desc: "" },
    { rate: "20%", acc: "63.0%", desc: "" },
    { rate: "25%", acc: "60.5%", desc: "" },
  ];

  const mechanism = [
    "Attacker corrupts training labels",
    "Model learns incorrect boundaries",
    "All future predictions affected",
  ];

  return (
    <div className="relative flex-1 flex flex-col p-8 overflow-hidden">

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-4xl font-display font-bold text-neon-secondary mb-2 leading-tight"
      >
        Causative Attack (Label Flipping)
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="relative z-10 text-lg text-neon-muted mb-6"
      >
        Poisons training data to corrupt learned model
      </motion.p>

      <div className="relative z-10 flex-1 grid grid-cols-1 lg:grid-cols-2 gap-8 min-h-0">
        <div className="flex flex-col gap-4 min-h-0">
          <NeonCard delay={0.4} glow="magenta" title="Attack Mechanism">
            <div className="mt-4 space-y-3">
              {mechanism.map((step, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.6 + i * 0.15 }}
                  className="flex items-center gap-4"
                >
                  <div className="w-8 h-8 shrink-0 rounded-full bg-neon-secondary/20 border border-neon-secondary/50 flex items-center justify-center text-neon-secondary font-bold">
                    {i + 1}
                  </div>
                  <span className="text-neon-text whitespace-normal">{step}</span>
                </motion.div>
              ))}
            </div>
          </NeonCard>

          <NeonCard delay={0.8} glow="violet" title="Poison Rate Impact">
            <div className="mt-4 overflow-x-auto custom-scrollbar">
              <table className="w-full min-w-[250px]">
                <thead>
                  <tr className="text-neon-muted text-sm border-b border-neon-muted/20">
                    <th className="text-left py-2 whitespace-nowrap">Poison Rate</th>
                    <th className="text-right py-2 whitespace-nowrap">Accuracy</th>
                  </tr>
                </thead>
                <tbody>
                  {poisonResults.map((r, i) => (
                    <motion.tr
                      key={r.rate}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 1.0 + i * 0.1 }}
                      className="border-b border-neon-muted/10"
                    >
                      <td className="py-2 text-neon-text font-mono">
                        {r.rate} {r.desc && <span className="text-neon-muted text-sm">({r.desc})</span>}
                      </td>
                      <td className="py-2 text-right font-mono text-neon-accent font-bold">{r.acc}</td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </NeonCard>
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.8 }}
          className="flex flex-col gap-4 min-h-0"
        >
          <div className="flex-1 min-h-0 rounded-xl overflow-hidden border-2 border-neon-secondary/30 bg-neon-bg/40 flex items-center justify-center p-4">
            <img
              src="/figures/causative_attack_boundaries.png"
              alt="Causative Attack Boundaries"
              className="max-w-full max-h-full object-contain"
            />
          </div>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.4 }}
            className="text-center text-neon-muted text-sm shrink-0"
          >
            Decision boundary shifts as poison rate increases
          </motion.p>
        </motion.div>
      </div>
    </div>
  );
}
