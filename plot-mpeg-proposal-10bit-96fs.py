import matplotlib.pyplot as plt
import math
import numpy

# bpp1_basketball = [0.114910364, 0.133041795, 0.161169455, 0.205691316, 0.273320164]
# psnr1_basketball = [75.29292877, 76.06785031, 76.63706656, 77.42059232, 78.25039306]
# # concate
#
# bpp2_basketball = [0.105415712, 0.176516128, 0.266805889, 0.486308301]
# psnr2_basketball = [71.10835861, 71.75797179, 72.62582756, 73.78195576]
# # vpcc v21 hm_ld 96frames
#
# bpp3_basketball = [0.103532896, 0.174536968, 0.237377121, 0.362379441]
# psnr3_basketball = [71.17073962, 71.84221092, 72.51118128, 73.25969942]
# # vpcc v21 hm_ra 96frames
#
# bpp4_basketball = [0.087013739, 0.138452028, 0.194393466, 0.302746312]
# psnr4_basketball = [71.4663065, 72.1383266, 72.83431319, 73.5818878]
# # vpcc v21 vvlib_slow_ra 96frames
#
# # 222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
# bpp1_dancer = [0.121292754, 0.140313536, 0.169373467, 0.214858838, 0.284455376]
# psnr1_dancer = [74.76182063, 75.49248445, 76.03957906, 76.8176418, 77.69554571]
# # concate
#
# bpp2_dancer = [0.131227361, 0.216138476, 0.319236323, 0.556819118]
# psnr2_dancer = [70.85326454, 71.49993498, 72.29637663, 73.31484996]
# # vpcc v21 hm_ld 96frames
#
# bpp3_dancer = [0.130751599, 0.216290461, 0.290556646, 0.430871243]
# psnr3_dancer = [70.88916003, 71.52795346, 72.14984861, 72.85148846]
# # vpcc v21 hm_ra 96frames
#
# bpp4_dancer = [0.111030519, 0.173103633, 0.239402953, 0.361214232]
# psnr4_dancer = [71.16070476, 71.80216055, 72.45653467, 73.1281015]
# # vpcc v21 vvlib_slow_ra 96frames
#
# # 333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
# bpp1_exercise = [0.113742787, 0.130588275, 0.157966, 0.202311916, 0.270181829]
# psnr1_exercise = [75.82895909, 76.45741388, 76.95004617, 77.72402052, 78.48452634]
# # concate
#
# bpp2_exercise = [0.101963974, 0.172561613, 0.26222351, 0.488704139]
# psnr2_exercise = [71.10871052, 71.71764188, 72.59535979, 73.79916329]
# # vpcc v21 hm_ld 96frames
#
# bpp3_exercise = [0.101765234, 0.171236261, 0.231789435, 0.358547151]
# psnr3_exercise = [71.19700426, 71.831579, 72.50514453, 73.26425934]
# # vpcc v21 hm_ra 96frames
#
# bpp4_exercise = [0.084410267, 0.134439869, 0.189017673, 0.297964385]
# psnr4_exercise = [71.45079598, 72.1359804, 72.82819699, 73.58698686]
# # vpcc v21 vvlib_slow_ra 96frames
#
# # 44444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444
# bpp1_model = [0.119806964, 0.138296819, 0.167333449, 0.214401804, 0.283686564]
# psnr1_model = [74.66704777, 75.48133358, 76.03008178, 76.81262666, 77.63056582]
# # concate
#
# bpp2_model = [0.123447734, 0.20534944, 0.314882737, 0.56479018]
# psnr2_model = [70.56716648, 71.02534717, 71.77336015, 72.74664117]
# # vpcc v21 hm_ld 96frames
#
# bpp3_model = [0.122263339, 0.204134309, 0.284282057, 0.43182583]
# psnr3_model = [70.61183606, 71.08840737, 71.66065463, 72.31603001]
# # vpcc v21 hm_ra 96frames
#
# bpp4_model = [0.103979794, 0.163931476, 0.232197392, 0.358109339]
# psnr4_model = [70.84297921, 71.31923998, 71.91061374, 72.54355196]
# # vpcc v21 vvlib_slow_ra 96frames

import pandas as pd
root_dir = './mpeg-results-96-10bit/'
df = pd.read_csv(root_dir + 'basketball.csv')
bpp1_basketball, psnr1_basketball = list(df['bpp']), list(df['d1-psnr'])
del bpp1_basketball[6]
del psnr1_basketball[6]
df = pd.read_csv(root_dir + 'dancer.csv')
bpp1_dancer, psnr1_dancer = list(df['bpp']), list(df['d1-psnr'])
del bpp1_dancer[6]
del psnr1_dancer[6]
df = pd.read_csv(root_dir + 'exercise.csv')
bpp1_exercise, psnr1_exercise = list(df['bpp']), list(df['d1-psnr'])
del bpp1_exercise[6]
del psnr1_exercise[6]
df = pd.read_csv(root_dir + 'model.csv')
bpp1_model, psnr1_model = list(df['bpp']), list(df['d1-psnr'])
del bpp1_model[6]
del psnr1_model[6]

bpp2_basketball = [0.048767258, 0.059335892, 0.075135096, 0.105415712, 0.176516128, 0.266805889, 0.486308301]
psnr2_basketball = [66.86005645, 68.62171779, 69.91483283, 71.10835861, 71.75797179, 72.62582756, 73.78195576]
# vpcc v21 hm_ld 96frames

bpp3_basketball = [0.04771883, 0.058351834, 0.073624405, 0.103532896, 0.174536968, 0.237377121, 0.362379441]
psnr3_basketball = [66.98420165, 68.7157745, 70.01418609, 71.17073962, 71.84221092, 72.51118128, 73.25969942]
# vpcc v21 hm_ra 96frames

bpp4_basketball = [0.0373838, 0.047253622, 0.062730136, 0.087013739, 0.138452028, 0.194393466, 0.302746312]
psnr4_basketball = [67.36561324, 69.02398757, 70.41591038, 71.4663065, 72.1383266, 72.83431319, 73.5818878]
# vpcc v21 vvlib_slow_ra 96frames

# 222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
bpp2_dancer = [0.058528302, 0.072495549, 0.093500303, 0.131227361, 0.216138476, 0.319236323, 0.556819118]
psnr2_dancer = [66.58057489, 68.34320501, 69.6855811, 70.85326454, 71.49993498, 72.29637663, 73.31484996]
# vpcc v21 hm_ld 96frames

bpp3_dancer = [0.058274916, 0.072443197, 0.093298914, 0.130751599, 0.216290461, 0.290556646, 0.430871243]
psnr3_dancer = [66.63683527, 68.40226786, 69.74095579, 70.88916003, 71.52795346, 72.14984861, 72.85148846]
# vpcc v21 hm_ra 96frames

bpp4_dancer = [0.047663945, 0.060407769, 0.080155733, 0.111030519, 0.173103633, 0.239402953, 0.361214232]
psnr4_dancer = [67.05498895, 68.7127931, 70.09083034, 71.16070476, 71.80216055, 72.45653467, 73.1281015]
# vpcc v21 vvlib_slow_ra 96frames

# 333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
bpp2_exercise = [0.046271971, 0.056846201, 0.071990074, 0.101963974, 0.172561613, 0.26222351, 0.488704139]
psnr2_exercise = [66.80841721, 68.57721466, 69.9083865, 71.10871052, 71.71764188, 72.59535979, 73.79916329]
# vpcc v21 hm_ld 96frames

bpp3_exercise = [0.045906089, 0.056542385, 0.071007031, 0.101765234, 0.171236261, 0.231789435, 0.358547151]
psnr3_exercise = [66.93643682, 68.70038333, 70.02547214, 71.19700426, 71.831579, 72.50514453, 73.26425934]
# vpcc v21 hm_ra 96frames

bpp4_exercise = [0.03541612, 0.04512956, 0.060342272, 0.084410267, 0.134439869, 0.189017673, 0.297964385]
psnr4_exercise = [67.20596083, 68.94739582, 70.37436144, 71.45079598, 72.1359804, 72.82819699, 73.58698686]
# vpcc v21 vvlib_slow_ra 96frames

# 44444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444
bpp2_model = [0.054287149, 0.067552324, 0.087101392, 0.123447734, 0.20534944, 0.314882737, 0.56479018]
psnr2_model = [66.51320747, 68.20307956, 69.45363435, 70.56716648, 71.02534717, 71.77336015, 72.74664117]
# vpcc v21 hm_ld 96frames

bpp3_model = [0.053295745, 0.066592371, 0.085714475, 0.122263339, 0.204134309, 0.284282057, 0.43182583]
psnr3_model = [66.63698843, 68.3097667, 69.54824533, 70.61183606, 71.08840737, 71.66065463, 72.31603001]
# vpcc v21 hm_ra 96frames

bpp4_model = [0.0429227, 0.054817518, 0.073920204, 0.103979794, 0.163931476, 0.232197392, 0.358109339]
psnr4_model = [66.99021778, 68.59112249, 69.8731772, 70.84297921, 71.31923998, 71.91061374, 72.54355196]
# vpcc v21 vvlib_slow_ra 96frames


# ##############################################################################
plt.rcParams.update({'font.size': 14.5})

plt.figure(figsize=(12, 12), dpi=600)
plt.subplot(2, 2, 1)
plt.title('Basketball-10bit D1-PSNR', size=19)
plt.xlabel('Rate/bpp', size=17)
plt.ylabel('D1-PSNR/dB', size=17)
plt.plot(bpp1_basketball, psnr1_basketball, marker='o', markersize=4)
plt.plot(bpp2_basketball, psnr2_basketball, marker='o', markersize=4)
plt.plot(bpp3_basketball, psnr3_basketball, marker='o', markersize=4)
plt.plot(bpp4_basketball, psnr4_basketball, marker='o', markersize=4)
plt.legend(['Proposed', 'hm_ld', 'hm_ra', 'vvlib_slow_ra'], loc='lower right')
plt.grid()

plt.subplot(2, 2, 2)
plt.subplots_adjust(wspace=0.225)
plt.title('Dancer-10bit D1-PSNR', size=19)
plt.xlabel('Rate/bpp', size=17)
plt.ylabel('D1-PSNR/dB', size=17)
plt.plot(bpp1_dancer, psnr1_dancer, marker='o', markersize=4)
plt.plot(bpp2_dancer, psnr2_dancer, marker='o', markersize=4)
plt.plot(bpp3_dancer, psnr3_dancer, marker='o', markersize=4)
plt.plot(bpp4_dancer, psnr4_dancer, marker='o', markersize=4)
plt.legend(['Proposed', 'hm_ld', 'hm_ra', 'vvlib_slow_ra'], loc='lower right')
plt.grid()

plt.subplot(2, 2, 3)
plt.title('Exercise-10bit D1-PSNR', size=19)
plt.xlabel('Rate/bpp', size=17)
plt.ylabel('D1-PSNR/dB', size=17)
plt.plot(bpp1_exercise, psnr1_exercise, marker='o', markersize=4)
plt.plot(bpp2_exercise, psnr2_exercise, marker='o', markersize=4)
plt.plot(bpp3_exercise, psnr3_exercise, marker='o', markersize=4)
plt.plot(bpp4_exercise, psnr4_exercise, marker='o', markersize=4)
plt.legend(['Proposed', 'hm_ld', 'hm_ra', 'vvlib_slow_ra'], loc='lower right')
plt.grid()

plt.subplot(2, 2, 4)
plt.subplots_adjust(wspace=0.225)
plt.title('Model-10bit D1-PSNR', size=19)
plt.xlabel('Rate/bpp', size=17)
plt.ylabel('D1-PSNR/dB', size=17)
plt.plot(bpp1_model, psnr1_model, marker='o', markersize=4)
plt.plot(bpp2_model, psnr2_model, marker='o', markersize=4)
plt.plot(bpp3_model, psnr3_model, marker='o', markersize=4)
plt.plot(bpp4_model, psnr4_model, marker='o', markersize=4)
plt.legend(['Proposed', 'hm_ld', 'hm_ra', 'vvlib_slow_ra'], loc='lower right')
plt.grid()


plt.tight_layout()
# plt.savefig('proposal.pdf', bbox_inches='tight')
plt.savefig('proposal-10bit-96frames.png', bbox_inches='tight')
# plt.show()



# --------------------- bdrate ---------------------
def bdsnr(metric_set1, metric_set2):
  """
  BJONTEGAARD    Bjontegaard metric calculation
  Bjontegaard's metric allows to compute the average gain in psnr between two
  rate-distortion curves [1].
  rate1,psnr1 - RD points for curve 1
  rate2,psnr2 - RD points for curve 2

  returns the calculated Bjontegaard metric 'dsnr'

  code adapted from code written by : (c) 2010 Giuseppe Valenzise
  http://www.mathworks.com/matlabcentral/fileexchange/27798-bjontegaard-metric/content/bjontegaard.m
  """
  # pylint: disable=too-many-locals
  # numpy seems to do tricks with its exports.
  # pylint: disable=no-member
  # map() is recommended against.
  # pylint: disable=bad-builtin
  rate1 = [x[0] for x in metric_set1]
  psnr1 = [x[1] for x in metric_set1]
  rate2 = [x[0] for x in metric_set2]
  psnr2 = [x[1] for x in metric_set2]

  log_rate1 = list(map(math.log, rate1))
  log_rate2 = list(map(math.log, rate2))

  # Best cubic poly fit for graph represented by log_ratex, psrn_x.
  poly1 = numpy.polyfit(log_rate1, psnr1, 3)
  poly2 = numpy.polyfit(log_rate2, psnr2, 3)

  # Integration interval.
  min_int = max([min(log_rate1), min(log_rate2)])
  max_int = min([max(log_rate1), max(log_rate2)])

  # Integrate poly1, and poly2.
  p_int1 = numpy.polyint(poly1)
  p_int2 = numpy.polyint(poly2)

  # Calculate the integrated value over the interval we care about.
  int1 = numpy.polyval(p_int1, max_int) - numpy.polyval(p_int1, min_int)
  int2 = numpy.polyval(p_int2, max_int) - numpy.polyval(p_int2, min_int)

  # Calculate the average improvement.
  if max_int != min_int:
    avg_diff = (int2 - int1) / (max_int - min_int)
  else:
    avg_diff = 0.0
  return avg_diff


def bdrate(metric_set1, metric_set2):
  """
  BJONTEGAARD    Bjontegaard metric calculation
  Bjontegaard's metric allows to compute the average % saving in bitrate
  between two rate-distortion curves [1].

  rate1,psnr1 - RD points for curve 1
  rate2,psnr2 - RD points for curve 2

  adapted from code from: (c) 2010 Giuseppe Valenzise

  """
  # numpy plays games with its exported functions.
  # pylint: disable=no-member
  # pylint: disable=too-many-locals
  # pylint: disable=bad-builtin
  rate1 = [x[0] for x in metric_set1]
  psnr1 = [x[1] for x in metric_set1]
  rate2 = [x[0] for x in metric_set2]
  psnr2 = [x[1] for x in metric_set2]

  log_rate1 = list(map(math.log, rate1))
  log_rate2 = list(map(math.log, rate2))

  # Best cubic poly fit for graph represented by log_ratex, psrn_x.
  poly1 = numpy.polyfit(psnr1, log_rate1, 3)
  poly2 = numpy.polyfit(psnr2, log_rate2, 3)

  # Integration interval.
  min_int = max([min(psnr1), min(psnr2)])
  max_int = min([max(psnr1), max(psnr2)])

  # find integral
  p_int1 = numpy.polyint(poly1)
  p_int2 = numpy.polyint(poly2)

  # Calculate the integrated value over the interval we care about.
  int1 = numpy.polyval(p_int1, max_int) - numpy.polyval(p_int1, min_int)
  int2 = numpy.polyval(p_int2, max_int) - numpy.polyval(p_int2, min_int)

  # Calculate the average improvement.
  avg_exp_diff = (int2 - int1) / (max_int - min_int)

  # In really bad formed data the exponent can grow too large.
  # clamp it.
  if avg_exp_diff > 200:
    avg_exp_diff = 200

  # Convert to a percentage.
  avg_diff = (math.exp(avg_exp_diff) - 1) * 100
  return avg_diff


avg_bd2 = 0
avg_bd3 = 0
avg_bd4 = 0

print('---------basketball---------')
bpp2 = bpp2_basketball
psnr2 = psnr2_basketball
bpp1 = bpp1_basketball
psnr1 = psnr1_basketball
metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
temp = bdrate(metric2, metric1)
print(temp)
avg_bd2 += temp

bpp2 = bpp3_basketball
psnr2 = psnr3_basketball
metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
temp = bdrate(metric2, metric1)
print(temp)
avg_bd3 += temp

bpp2 = bpp4_basketball
psnr2 = psnr4_basketball
metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
temp = bdrate(metric2, metric1)
print(temp)
avg_bd4 += temp

print('---------dancer---------')
bpp2 = bpp2_dancer
psnr2 = psnr2_dancer
bpp1 = bpp1_dancer
psnr1 = psnr1_dancer
metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
temp = bdrate(metric2, metric1)
print(temp)
avg_bd2 += temp

bpp2 = bpp3_dancer
psnr2 = psnr3_dancer
metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
temp = bdrate(metric2, metric1)
print(temp)
avg_bd3 += temp

bpp2 = bpp4_dancer
psnr2 = psnr4_dancer
metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
temp = bdrate(metric2, metric1)
print(temp)
avg_bd4 += temp

print('---------exercise---------')
bpp2 = bpp2_exercise
psnr2 = psnr2_exercise
bpp1 = bpp1_exercise
psnr1 = psnr1_exercise
metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
temp = bdrate(metric2, metric1)
print(temp)
avg_bd2 += temp

bpp2 = bpp3_exercise
psnr2 = psnr3_exercise
metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
temp = bdrate(metric2, metric1)
print(temp)
avg_bd3 += temp

bpp2 = bpp4_exercise
psnr2 = psnr4_exercise
metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
temp = bdrate(metric2, metric1)
print(temp)
avg_bd4 += temp

print('---------model---------')
bpp2 = bpp2_model
psnr2 = psnr2_model
bpp1 = bpp1_model
psnr1 = psnr1_model
metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
temp = bdrate(metric2, metric1)
print(temp)
avg_bd2 += temp

bpp2 = bpp3_model
psnr2 = psnr3_model
metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
temp = bdrate(metric2, metric1)
print(temp)
avg_bd3 += temp

bpp2 = bpp4_model
psnr2 = psnr4_model
metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
temp = bdrate(metric2, metric1)
print(temp)
avg_bd4 += temp

print('---------average---------')
print('average-d1', avg_bd2/4, avg_bd3/4, avg_bd4/4)



# avg_bd2 = 0
# avg_bd3 = 0
# avg_bd4 = 0
#
# print('---------basketball---------')
# bpp2 = bpp2_basketball
# psnr2 = psnr2d2_basketball
# bpp1 = bpp1_basketball
# psnr1 = psnr1d2_basketball
# # bpp1 = bpp1_basketball
# # psnr1 = psnr1d2_basketball
# metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
# metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
# temp = bdrate(metric2, metric1)
# print(temp)
# avg_bd2 += temp
#
# bpp2 = bpp3_basketball
# psnr2 = psnr3d2_basketball
# metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
# metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
# temp = bdrate(metric2, metric1)
# print(temp)
# avg_bd3 += temp
#
# bpp2 = bpp4_basketball
# psnr2 = psnr4d2_basketball
# metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
# metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
# temp = bdrate(metric2, metric1)
# print(temp)
# avg_bd4 += temp
#
# print('---------dancer---------')
# bpp2 = bpp2_dancer
# psnr2 = psnr2d2_dancer
# bpp1 = bpp1_dancer
# psnr1 = psnr1d2_dancer
# metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
# metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
# temp = bdrate(metric2, metric1)
# print(temp)
# avg_bd2 += temp
#
# bpp2 = bpp3_dancer
# psnr2 = psnr3d2_dancer
# metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
# metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
# temp = bdrate(metric2, metric1)
# print(temp)
# avg_bd3 += temp
#
# bpp2 = bpp4_dancer
# psnr2 = psnr4d2_dancer
# metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
# metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
# temp = bdrate(metric2, metric1)
# print(temp)
# avg_bd4 += temp
#
# print('---------exercise---------')
# bpp2 = bpp2_exercise
# psnr2 = psnr2d2_exercise
# bpp1 = bpp1_exercise
# psnr1 = psnr1d2_exercise
# metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
# metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
# temp = bdrate(metric2, metric1)
# print(temp)
# avg_bd2 += temp
#
# bpp2 = bpp3_exercise
# psnr2 = psnr3d2_exercise
# metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
# metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
# temp = bdrate(metric2, metric1)
# print(temp)
# avg_bd3 += temp
#
# bpp2 = bpp4_exercise
# psnr2 = psnr4d2_exercise
# metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
# metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
# temp = bdrate(metric2, metric1)
# print(temp)
# avg_bd4 += temp
#
# print('---------model---------')
# bpp2 = bpp2_model
# psnr2 = psnr2d2_model
# bpp1 = bpp1_model
# psnr1 = psnr1d2_model
# metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
# metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
# temp = bdrate(metric2, metric1)
# print(temp)
# avg_bd2 += temp
#
# bpp2 = bpp3_model
# psnr2 = psnr3d2_model
# metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
# metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
# temp = bdrate(metric2, metric1)
# print(temp)
# avg_bd3 += temp
#
# bpp2 = bpp4_model
# psnr2 = psnr4d2_model
# metric1=[(bpp1[i], psnr1[i]) for i in range(len(bpp1))]
# metric2 = [(bpp2[i], psnr2[i]) for i in range(len(bpp2))]
# temp = bdrate(metric2, metric1)
# print(temp)
# avg_bd4 += temp
#
# print('---------average---------')
# print('average-d2', avg_bd2/4, avg_bd3/4, avg_bd4/4)