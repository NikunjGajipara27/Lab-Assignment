import re
import heapq
from typing import List, Tuple, Dict

def preprocess_text(text: str) -> List[str]:
   txt = text.strip().replace('\n', ' ')
   parts = re.split(r'(?<=[.!?])\s+', txt)
   sentences = []
   for p in parts:
       s = p.strip().lower()
       if s:
           sentences.append(s)
   return sentences

def levenshtein(a: str, b: str) -> int:
   la, lb = len(a), len(b)
   if la == 0: return lb
   if lb == 0: return la
   prev = list(range(lb + 1))
   cur = [0] * (lb + 1)
   for i in range(1, la + 1):
       cur[0] = i
       for j in range(1, lb + 1):
           cost = 0 if a[i - 1] == b[j - 1] else 1
           cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
       prev, cur = cur, prev
   return prev[lb]

def a_star_align(sentA: List[str], sentB: List[str], gap_cost: int = 5):
   m, n = len(sentA), len(sentB)
   lev_cache = {}
   def lev(i, j):
       if (i, j) not in lev_cache:
           lev_cache[(i, j)] = levenshtein(sentA[i], sentB[j])
       return lev_cache[(i, j)]

   open_heap = [(0, 0, 0, 0)]
   came_from, g_score = {}, {(0, 0): 0}
   while open_heap:
       f, g, i, j = heapq.heappop(open_heap)
       if (i, j) == (m, n):
           path, cur = [], (i, j)
           while cur != (0, 0):
               prev = came_from[cur]
               path.append((prev[0], prev[1], prev[2], prev[3], cur[0], cur[1], prev[4]))
               cur = (prev[0], prev[1])
           return g, list(reversed(path))
       if i < m and j < n:
           c = lev(i, j); ni, nj = i+1, j+1
           ng = g + c
           if ng < g_score.get((ni, nj), 1e9):
               g_score[(ni, nj)] = ng
               came_from[(ni, nj)] = (i, j, 'ALIGN', c, {'lev': c})
               heapq.heappush(open_heap, (ng, ng, ni, nj))
       if i < m:
           ni, nj = i+1, j; ng = g + gap_cost
           if ng < g_score.get((ni, nj), 1e9):
               g_score[(ni, nj)] = ng
               came_from[(ni, nj)] = (i, j, 'SKIP_A', gap_cost, None)
               heapq.heappush(open_heap, (ng, ng, ni, nj))
       if j < n:
           ni, nj = i, j+1; ng = g + gap_cost
           if ng < g_score.get((ni, nj), 1e9):
               g_score[(ni, nj)] = ng
               came_from[(ni, nj)] = (i, j, 'SKIP_B', gap_cost, None)
               heapq.heappush(open_heap, (ng, ng, ni, nj))
   return float('inf'), []

def detect_plagiarism(path, sentA, sentB, threshold=0.35):
   results = []
   for step in path:
       if step[2] == 'ALIGN':
           i, j = step[0], step[1]
           lev_val = step[6].get('lev', 0)
           la, lb = len(sentA[i].split()), len(sentB[j].split())
           ned = lev_val / max(1, max(la, lb))
           if ned <= threshold:
               results.append((i, j, sentA[i], sentB[j], lev_val, round(ned, 3)))
   return results

if __name__ == '__main__':
   docA = 'Artificial intelligence is the study of agents that perceive and act. Agents take actions in an environment.'
   docB = 'Artificial intelligence studies agents that perceive and act. Agents operate in an environment.'
   sA, sB = preprocess_text(docA), preprocess_text(docB)
   total_cost, path = a_star_align(sA, sB)
   print('Total Cost:', total_cost)
   print('Potential plagiarism pairs:', detect_plagiarism(path, sA, sB))