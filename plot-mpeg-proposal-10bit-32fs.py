import matplotlib.pyplot as plt
import math
import numpy

# bpp1_basketball = [0.111873125, 0.129749375, 0.1589459375, 0.2037796875, 0.2700921875]
# psnr1_basketball = [75.4857094, 76.2428562, 76.7960125, 77.586221875, 78.4194656]
# # concate
#
# bpp2_basketball = [0.096273408, 0.163595474, 0.251386305, 0.469764757]
# psnr2_basketball = [71.16644259, 71.78463043, 72.68401076, 73.9048878]
# # vpcc v21 hm_ld 32frames
#
# bpp3_basketball = [0.095010093, 0.161079366, 0.220136953, 0.342408441]
# psnr3_basketball = [71.27232836, 71.91262218, 72.59157765, 73.36632761]
# # vpcc v21 hm_ra 32frames
#
# bpp4_basketball = [0.078358057, 0.126261995, 0.179202931, 0.284767221]
# psnr4_basketball = [71.52141679, 72.20288239, 72.90657013, 73.69823391]
# # vpcc v21 vvlib_slow_ra 32frames
#
# # 222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
# bpp1_dancer = [0.1225759375, 0.14141125, 0.171844687, 0.217805625, 0.287611875]
# psnr1_dancer = [74.9194219, 75.696356, 76.262175, 77.074075, 77.888253]
# # concate
#
# bpp2_dancer = [0.128602295, 0.212018635, 0.314959235, 0.55208857]
# psnr2_dancer = [70.89601442, 71.57075399, 72.39870024, 73.45818973]
# # vpcc v21 hm_ld 32frames
#
# bpp3_dancer = [0.127446235, 0.212647176, 0.286869101, 0.426384963]
# psnr3_dancer = [70.92749088, 71.60429163, 72.24208636, 72.97126696]
# # vpcc v21 hm_ra 32frames
#
# bpp4_dancer = [0.108234289, 0.169195667, 0.234984398, 0.355748248]
# psnr4_dancer = [71.2144032, 71.88080731, 72.56098988, 73.25222709]
# # vpcc v21 vvlib_slow_ra 32frames
#
# # 333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
# bpp1_exercise = [0.111640625, 0.1276021875, 0.1568528125, 0.2011703125, 0.2687115625]
# psnr1_exercise = [75.8117, 76.46255, 76.994134375, 77.784434375, 78.563746875]
# # concate
#
# bpp2_exercise = [0.093045317, 0.159651118, 0.245806668, 0.470121192]
# psnr2_exercise = [71.18310518, 71.73411062, 72.63536422, 73.92376044]
# # vpcc v21 hm_ld 32frames
#
# bpp3_exercise = [0.09250657, 0.1574491, 0.215427684, 0.338767483]
# psnr3_exercise = [71.2865828, 71.89869029, 72.59455508, 73.37294107]
# # vpcc v21 hm_ra 32frames
#
# bpp4_exercise = [0.076999909, 0.124020402, 0.176171419, 0.280312477]
# psnr4_exercise = [71.52291824, 72.18716314, 72.89040378, 73.6872686]
# # vpcc v21 vvlib_slow_ra 32frames
#
# # 44444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444
# bpp1_model = [0.1175503125, 0.1367715625, 0.1660121875, 0.2133046875, 0.2819578125]
# psnr1_model = [74.793928125, 75.5783875, 76.17682499999998, 76.96748125, 77.81295625]
# # concate
#
# bpp2_model = [0.1109962, 0.188657689, 0.295486722, 0.547545953]
# psnr2_model = [70.55354014, 71.0236172, 71.81444962, 72.87925367]
# # vpcc v21 hm_ld 32frames
#
# bpp3_model = [0.110539432, 0.188238797, 0.265445393, 0.413259444]
# psnr3_model = [70.62123771, 71.11086788, 71.70327923, 72.40368508]
# # vpcc v21 hm_ra 32frames
#
# bpp4_model = [0.094036895, 0.150472908, 0.216682563, 0.342141593]
# psnr4_model = [70.81581256, 71.3140557, 71.93006889, 72.63763954]
# # vpcc v21 vvlib_slow_ra 32frames

import pandas as pd
root_dir = './mpeg-results-32-10bit/'
df = pd.read_csv(root_dir + 'basketball-32frames.csv')
bpp1_basketball, psnr1_basketball = df['bpp'], df['d1-psnr']
df = pd.read_csv(root_dir + 'dancer-32frames.csv')
bpp1_dancer, psnr1_dancer = df['bpp'], df['d1-psnr']
df = pd.read_csv(root_dir + 'exercise-32frames.csv')
bpp1_exercise, psnr1_exercise = df['bpp'], df['d1-psnr']
df = pd.read_csv(root_dir + 'model-32frames.csv')
bpp1_model, psnr1_model = df['bpp'], df['d1-psnr']


bpp2_basketball = [0.045026883, 0.054356984, 0.068159706, 0.096273408, 0.163595474, 0.251386305, 0.469764757]
psnr2_basketball = [66.98432254, 68.72934525, 70.00184932, 71.16644259, 71.78463043, 72.68401076, 73.9048878]
# vpcc v21 hm_ld 32frames

bpp3_basketball = [0.044372905, 0.053739675, 0.066482832, 0.095010093, 0.161079366, 0.220136953, 0.342408441]
psnr3_basketball = [67.20019073, 68.85018986, 70.10869568, 71.27232836, 71.91262218, 72.59157765, 73.36632761]
# vpcc v21 hm_ra 32frames

bpp4_basketball = [0.033218299, 0.042018458, 0.056016321, 0.078358057, 0.126261995, 0.179202931, 0.284767221]
psnr4_basketball = [67.44828712, 69.13248879, 70.48586514, 71.52141679, 72.20288239, 72.90657013, 73.69823391]
# vpcc v21 vvlib_slow_ra 32frames

# 222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
bpp2_dancer = [0.056786399, 0.07061466, 0.091533307, 0.128602295, 0.212018635, 0.314959235, 0.55208857]
psnr2_dancer = [66.55510052, 68.34469217, 69.7031193, 70.89601442, 71.57075399, 72.39870024, 73.45818973]
# vpcc v21 hm_ld 32frames

bpp3_dancer = [0.056351009, 0.070149035, 0.090857456, 0.127446235, 0.212647176, 0.286869101, 0.426384963]
psnr3_dancer = [66.61329485, 68.38880299, 69.7550031, 70.92749088, 71.60429163, 72.24208636, 72.97126696]
# vpcc v21 hm_ra 32frames

bpp4_dancer = [0.045710985, 0.058418045, 0.077927365, 0.108234289, 0.169195667, 0.234984398, 0.355748248]
psnr4_dancer = [67.03110947, 68.71995466, 70.1113298, 71.2144032, 71.88080731, 72.56098988, 73.25222709]
# vpcc v21 vvlib_slow_ra 32frames

# 333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
bpp2_exercise = [0.042575161, 0.052157516, 0.065442658, 0.093045317, 0.159651118, 0.245806668, 0.470121192]
psnr2_exercise = [66.97871544, 68.69603316, 69.99828499, 71.18310518, 71.73411062, 72.63536422, 73.92376044]
# vpcc v21 hm_ld 32frames

bpp3_exercise = [0.041916735, 0.051184407, 0.064164042, 0.09250657, 0.1574491, 0.215427684, 0.338767483]
psnr3_exercise = [67.15441717, 68.87339257, 70.14749156, 71.2865828, 71.89869029, 72.59455508, 73.37294107]
# vpcc v21 hm_ra 32frames

bpp4_exercise = [0.031495479, 0.040335289, 0.054356094, 0.076999909, 0.124020402, 0.176171419, 0.280312477]
psnr4_exercise = [67.37286958, 69.0439599, 70.44679695, 71.52291824, 72.18716314, 72.89040378, 73.6872686]
# vpcc v21 vvlib_slow_ra 32frames

# 44444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444
bpp2_model = [0.048736579, 0.060626806, 0.077085466, 0.1109962, 0.188657689, 0.295486722, 0.547545953]
psnr2_model = [66.50783343, 68.21015527, 69.42403257, 70.55354014, 71.0236172, 71.81444962, 72.87925367]
# vpcc v21 hm_ld 32frames

bpp3_model = [0.048096803, 0.059999781, 0.076257807, 0.110539432, 0.188238797, 0.265445393, 0.413259444]
psnr3_model = [66.74575046, 68.36197399, 69.55858603, 70.62123771, 71.11086788, 71.70327923, 72.40368508]
# vpcc v21 hm_ra 32frames

bpp4_model = [0.03753113, 0.048191307, 0.065546628, 0.094036895, 0.150472908, 0.216682563, 0.342141593]
psnr4_model = [66.998137, 68.60608024, 69.82650724, 70.81581256, 71.3140557, 71.93006889, 72.63763954]
# vpcc v21 vvlib_slow_ra 32frames


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
plt.savefig('proposal-10bit-32frames.png', bbox_inches='tight')
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