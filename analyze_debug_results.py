#!/usr/bin/env python3
"""
ANALYZE DEBUG RESULTS

This script analyzes the output logs from the debugging runs to help identify
the correct tokenization and logit positioning approach for each model.

Usage:
python analyze_debug_results.py
"""

import re
import os
from typing import Dict, List, Tuple
import json

class DebugResultsAnalyzer:
    def __init__(self, debug_dir: str = "debug_outputs"):
        self.debug_dir = debug_dir
        self.results = {}
    
    def parse_log_file(self, filepath: str) -> Dict:
        """Parse a debug log file and extract key metrics."""
        if not os.path.exists(filepath):
            return {"error": f"File not found: {filepath}"}
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract run summaries
        runs = {}
        
        # Look for RUN summaries
        run_pattern = r'📊 (RUN\d+) SUMMARY:\s*\n(.*?)(?=📊|🎯|$)'
        run_matches = re.findall(run_pattern, content, re.DOTALL)
        
        for run_name, summary_content in run_matches:
            run_data = {}
            
            # Extract matches/mismatches for CounterFact
            matches_pattern = r'Matches: (\d+)/(\d+)'
            matches_match = re.search(matches_pattern, summary_content)
            if matches_match:
                run_data['matches'] = int(matches_match.group(1))
                run_data['total'] = int(matches_match.group(2))
                run_data['match_rate'] = run_data['matches'] / run_data['total'] if run_data['total'] > 0 else 0
            
            # Extract mismatches
            mismatches_pattern = r'Mismatches: (\d+)/(\d+)'
            mismatches_match = re.search(mismatches_pattern, summary_content)
            if mismatches_match:
                run_data['mismatches'] = int(mismatches_match.group(1))
            
            # Extract average target probabilities
            avg_prob_pattern = r'Avg target prob \(matches\): ([\d.]+)'
            avg_prob_match = re.search(avg_prob_pattern, summary_content)
            if avg_prob_match:
                run_data['avg_target_prob'] = float(avg_prob_match.group(1))
            
            # Extract average NLL
            avg_nll_pattern = r'Avg target NLL \(matches\): ([\d.]+)'
            avg_nll_match = re.search(avg_nll_pattern, summary_content)
            if avg_nll_match:
                run_data['avg_target_nll'] = float(avg_nll_match.group(1))
            
            # Extract errors
            errors_pattern = r'Errors: (\d+)'
            errors_match = re.search(errors_pattern, summary_content)
            if errors_match:
                run_data['errors'] = int(errors_match.group(1))
            
            # For ZSRE: extract target-specific matches
            target_matches = {}
            target_pattern = r"'([^']+)' - Token matches: (\d+), Text matches: (\d+)"
            target_matches_list = re.findall(target_pattern, summary_content)
            for target, token_matches, text_matches in target_matches_list:
                target_matches[target] = {
                    'token_matches': int(token_matches),
                    'text_matches': int(text_matches)
                }
            if target_matches:
                run_data['target_matches'] = target_matches
            
            # Extract valid predictions count
            valid_pattern = r'Valid predictions: (\d+)/(\d+)'
            valid_match = re.search(valid_pattern, summary_content)
            if valid_match:
                run_data['valid_predictions'] = int(valid_match.group(1))
                run_data['total_predictions'] = int(valid_match.group(2))
                run_data['valid_rate'] = run_data['valid_predictions'] / run_data['total_predictions'] if run_data['total_predictions'] > 0 else 0
            
            runs[run_name] = run_data
        
        return runs
    
    def analyze_all_logs(self):
        """Analyze all debug log files."""
        log_files = [
            ("llama_counterfact", "llama_counterfact_debug.log"),
            ("llama_zsre", "llama_zsre_debug.log"),
            ("deepseek_counterfact", "deepseek_counterfact_debug.log"),
            ("deepseek_zsre", "deepseek_zsre_debug.log")
        ]
        
        for key, filename in log_files:
            filepath = os.path.join(self.debug_dir, filename)
            self.results[key] = self.parse_log_file(filepath)
    
    def find_best_run(self, runs_data: Dict) -> Tuple[str, Dict]:
        """Find the best performing run based on multiple criteria."""
        if not runs_data or "error" in runs_data:
            return None, None
        
        best_run = None
        best_score = -1
        
        for run_name, run_data in runs_data.items():
            if not run_data:
                continue
                
            score = 0
            
            # For CounterFact: prioritize match rate and avg target prob
            if 'match_rate' in run_data:
                score += run_data['match_rate'] * 50  # Weight match rate heavily
                
            if 'avg_target_prob' in run_data:
                score += run_data['avg_target_prob'] * 30  # Weight probability
                
            if 'errors' in run_data:
                score -= run_data['errors'] * 10  # Penalize errors
            
            # For ZSRE: prioritize valid predictions and target matches
            if 'valid_rate' in run_data:
                score += run_data['valid_rate'] * 40
                
            if 'target_matches' in run_data:
                total_token_matches = sum(tm['token_matches'] for tm in run_data['target_matches'].values())
                total_text_matches = sum(tm['text_matches'] for tm in run_data['target_matches'].values())
                score += (total_token_matches + total_text_matches) * 2
            
            if score > best_score:
                best_score = score
                best_run = run_name
        
        return best_run, runs_data.get(best_run, {})
    
    def generate_report(self):
        """Generate a comprehensive analysis report."""
        print("🔍 DEBUG RESULTS ANALYSIS REPORT")
        print("=" * 80)
        
        for analysis_key, runs_data in self.results.items():
            model_name = "Llama" if "llama" in analysis_key else "DeepSeek"
            task_name = "CounterFact" if "counterfact" in analysis_key else "ZSRE"
            
            print(f"\n📊 {model_name} - {task_name}")
            print("-" * 40)
            
            if "error" in runs_data:
                print(f"❌ Error: {runs_data['error']}")
                continue
            
            if not runs_data:
                print("❌ No data found")
                continue
            
            # Show results for each run
            for run_name in ["RUN1", "RUN2", "RUN3"]:
                if run_name in runs_data:
                    run_data = runs_data[run_name]
                    print(f"\n  {run_name}:")
                    
                    if 'match_rate' in run_data:
                        print(f"    Match rate: {run_data['match_rate']:.2%} ({run_data['matches']}/{run_data['total']})")
                    
                    if 'avg_target_prob' in run_data:
                        print(f"    Avg target prob: {run_data['avg_target_prob']:.4f}")
                    
                    if 'avg_target_nll' in run_data:
                        print(f"    Avg target NLL: {run_data['avg_target_nll']:.4f}")
                    
                    if 'valid_rate' in run_data:
                        print(f"    Valid predictions: {run_data['valid_rate']:.2%} ({run_data['valid_predictions']}/{run_data['total_predictions']})")
                    
                    if 'errors' in run_data:
                        print(f"    Errors: {run_data['errors']}")
                    
                    if 'target_matches' in run_data:
                        print(f"    Target matches:")
                        for target, matches in run_data['target_matches'].items():
                            print(f"      '{target}': {matches['token_matches']} token, {matches['text_matches']} text")
            
            # Find and highlight best run
            best_run, best_data = self.find_best_run(runs_data)
            if best_run:
                print(f"\n  🏆 RECOMMENDED: {best_run}")
                if 'match_rate' in best_data:
                    print(f"    Best match rate: {best_data['match_rate']:.2%}")
                if 'avg_target_prob' in best_data:
                    print(f"    Best avg prob: {best_data['avg_target_prob']:.4f}")
        
        # Generate implementation recommendations
        print(f"\n🛠️  IMPLEMENTATION RECOMMENDATIONS")
        print("=" * 80)
        
        recommendations = {}
        
        for analysis_key, runs_data in self.results.items():
            if "error" not in runs_data and runs_data:
                model_name = "llama" if "llama" in analysis_key else "deepseek"
                task_name = "counterfact" if "counterfact" in analysis_key else "zsre"
                
                best_run, _ = self.find_best_run(runs_data)
                if best_run:
                    if model_name not in recommendations:
                        recommendations[model_name] = {}
                    recommendations[model_name][task_name] = best_run
        
        for model, tasks in recommendations.items():
            print(f"\n{model.upper()} MODEL:")
            for task, best_run in tasks.items():
                if best_run == "RUN1":
                    formula = "logit_idx = padding_offset + prefix_lens[prefix_idx] + j"
                    logits_adj = "No logits adjustment"
                elif best_run == "RUN2":
                    formula = "logit_idx = padding_offset + prefix_lens[prefix_idx] + j - 1"
                    logits_adj = "No logits adjustment"
                else:  # RUN3
                    formula = "logit_idx = padding_offset + prefix_lens[prefix_idx] + j - 1"
                    logits_adj = "logits = logits[:, 1:, :]"
                
                print(f"  {task.upper()}: Use {best_run}")
                print(f"    Formula: {formula}")
                print(f"    Logits: {logits_adj}")
        
        print(f"\n📝 NEXT STEPS:")
        print("1. Review the detailed results above")
        print("2. Update your eval_utils_counterfact.py and eval_utils_zsre.py")
        print("3. Test with actual model editing to confirm improvements")
        print("4. Look for consistent patterns in model predictions")

    def save_results(self, filepath: str = "debug_analysis_results.json"):
        """Save analysis results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to {filepath}")


def main():
    analyzer = DebugResultsAnalyzer()
    analyzer.analyze_all_logs()
    analyzer.generate_report()
    analyzer.save_results()


if __name__ == "__main__":
    main()