"use client";

import { motion } from "framer-motion";
import NeonCard from "@/components/NeonCard";
import FormulaBlock from "@/components/FormulaBlock";

export default function Slide09FGSMAttack() {
  const epsilonResults = [
    { epsilon: 0.01, robust: "32.1%" },
    { epsilon: 0.05, robust: "17.7%" },
    { epsilon: 0.10, robust: "13.3%" },
    { epsilon: 0.50, robust: "3.1%" },
  ];

  return (
    <div className="relative flex-1 flex flex-col p-8 overflow-hidden">

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-4xl font-display font-bold text-neon-secondary mb-2 leading-tight"
      >
        Exploratory Attack (FGSM)
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="relative z-10 text-lg text-neon-muted mb-4"
      >
        Perturbs test-time inputs to evade detection
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="relative z-10 mb-4 shrink-0"
      >
        <FormulaBlock
          formula="x_adv = x + ε · sign(∇_x J(θ, x, y))"
          description="Adversarial example computation"
          glow="magenta"
        />
      </motion.div>

      <div className="relative z-10 flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 min-h-0">
        <div className="flex flex-col gap-3 min-h-0">
          <NeonCard delay={0.6} glow="magenta" title="Attack Results" className="shrink-0">
            <div className="mt-4 overflow-x-auto custom-scrollbar">
              <table className="w-full min-w-[250px]">
                <thead>
                  <tr className="text-neon-muted text-sm border-b border-neon-muted/20">
                    <th className="text-left py-1.5 whitespace-nowrap">ε</th>
                    <th className="text-right py-1.5 whitespace-nowrap">Robust Acc.</th>
                  </tr>
                </thead>
                <tbody>
                  {epsilonResults.map((r, i) => (
                    <motion.tr
                      key={r.epsilon}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.8 + i * 0.1 }}
                      className="border-b border-neon-muted/10"
                    >
                      <td className="py-2 text-neon-text font-mono text-sm">{r.epsilon.toFixed(2)}</td>
                      <td className="py-2 text-right font-mono text-neon-secondary font-bold text-sm">{r.robust}</td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </NeonCard>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.2 }}
            className="space-y-2 shrink-0"
          >
            <div className="p-2.5 bg-neon-bg/60 border border-neon-muted/30 rounded-lg">
              <span className="text-neon-muted text-sm">Model: </span>
              <span className="text-neon-text font-bold text-sm">Linear SVM (Astute: 90.2%)</span>
            </div>
            <div className="p-2.5 bg-neon-bg/60 border border-neon-muted/30 rounded-lg">
              <span className="text-neon-muted text-sm">Target: </span>
              <span className="text-neon-text font-bold text-sm">Test samples only</span>
            </div>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.8 }}
          className="rounded-xl overflow-hidden border-2 border-neon-secondary/30 bg-neon-bg/40 flex items-center justify-center p-4 min-h-0"
        >
          <img
            src="/figures/fgsm_attack_analysis.png"
            alt="FGSM Attack Analysis"
            className="max-w-full max-h-full object-contain"
          />
        </motion.div>
      </div>
    </div>
  );
}
