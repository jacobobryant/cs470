import asyncio
import json
from time import sleep
import numpy as np
import math
from functools import reduce

debug = False

# robot is pointed towards the target
example1 = {"time": 2189617.79221862, "21": {"corners": [[991.0, 478.0], [1009.0, 573.0], [912.0, 591.0], [894.0, 497.0]],
                                             "orientation": [0.9822942018508911, -0.1873447746038437], "center": [951.5, 534.75]},
        "robot": {"corners": [[973.0, 263.0], [867.0, 288.0], [842.0, 180.0], [949.0, 156.0]], "orientation": [0.22220909595489502, 0.9749990701675415], "center": [907.75, 221.75]}}

# robot is pointed away from the target
example2 = {"time": 2189691.664889238, "21": {"corners": [[990.0, 478.0], [1009.0, 573.0], [913.0, 592.0], [894.0, 496.0]],
                                              "orientation": [0.9819334149360657, -0.18922674655914307], "center": [951.5, 534.75]},
        "robot": {"corners": [[902.0, 143.0], [1005.0, 182.0], [966.0, 285.0], [863.0, 245.0]], "orientation": [0.35561609268188477, -0.9346320629119873], "center": [934.0, 213.75]}}

# target is on left side of robot
example3 = {"robot": {"corners": [[787.0, 208.0], [756.0, 107.0], [859.0, 75.0], [891.0, 177.0]], "center": [823.25, 141.75], "orientation": [-0.9566738605499268, 0.29116159677505493]}, "21": {"corners": [[932.0, 390.0], [967.0, 478.0], [876.0, 514.0], [842.0, 425.0]], "center": [904.25, 451.75], "orientation": [0.9309388995170593, -0.36517491936683655]}, "time": 2248780.235014935}

# target is on right side of robot
example4 = {"robot": {"corners": [[898.0, 68.0], [882.0, 173.0], [775.0, 157.0], [792.0, 52.0]], "center": [836.75, 112.5], "orientation": [0.98890221118927, 0.14856746792793274]}, "21": {"corners": [[932.0, 390.0], [967.0, 478.0], [877.0, 514.0], [841.0, 426.0]], "center": [904.25, 452.0], "orientation": [0.9291830658912659, -0.36961978673934937]}, "time": 2248804.190855203}

example5 = {"time": 17880.044430377, "robot": {"corners": [[1251.0, 314.0], [1174.0, 250.0], [1239.0, 177.0], [1317.0, 244.0]], "orientation": [-0.6754910945892334, 0.737368106842041], "center": [1245.25, 246.25]}, "29": {"corners": [[830.0, 417.0], [735.0, 410.0], [743.0, 318.0], [837.0, 323.0]], "orientation": [-0.08038419485092163, 0.9967640042304993], "center": [786.25, 367.0]}, "25": {"corners": [[400.0, 377.0], [377.0, 291.0], [466.0, 262.0], [490.0, 348.0]], "orientation": [-0.9513070583343506, 0.3082447350025177], "center": [433.25, 319.5]}}

# L-shaped maze. target is 29.
example6 = {"time": 676.712433488, "53": {"corners": [[1315.0, 380.0], [1302.0, 289.0], [1411.0, 286.0], [1426.0, 375.0]], "orientation": [-0.9993395209312439, 0.036339618265628815], "center": [1363.5, 332.5]}, "29": {"corners": [[1361.0, 235.0], [1257.0, 244.0], [1240.0, 162.0], [1341.0, 153.0]], "orientation": [0.22007830440998077, 0.9754822254180908], "center": [1299.75, 198.5]}, "23": {"corners": [[1169.0, 857.0], [1176.0, 989.0], [1043.0, 990.0], [1040.0, 858.0]], "orientation": [0.9999708533287048, -0.00763336569070816], "center": [1107.0, 923.5]}, "robot": {"corners": [[729.0, 797.0], [873.0, 794.0], [871.0, 942.0], [719.0, 946.0]], "orientation": [0.04037109762430191, -0.9991847276687622], "center": [798.0, 869.75]}, "99": {"corners": [[466.0, 895.0], [480.0, 773.0], [602.0, 767.0], [592.0, 889.0]], "orientation": [-0.9988314509391785, 0.048330552875995636], "center": [535.0, 831.0]}, "37": {"corners": [[1164.0, 794.0], [1039.0, 793.0], [1038.0, 676.0], [1159.0, 676.0]], "orientation": [0.02552359737455845, 0.9996742010116577], "center": [1100.0, 734.75]}, "91": {"corners": [[502.0, 595.0], [615.0, 591.0], [604.0, 700.0], [487.0, 705.0]], "orientation": [0.1178935244679451, -0.9930262565612793], "center": [552.0, 647.75]}, "43": {"corners": [[1030.0, 512.0], [1143.0, 509.0], [1151.0, 614.0], [1035.0, 617.0]], "orientation": [-0.06178648769855499, -0.9980894327163696], "center": [1089.75, 563.0]}, "39": {"corners": [[630.0, 426.0], [616.0, 523.0], [505.0, 523.0], [523.0, 426.0]], "orientation": [1.0, 0.0], "center": [568.5, 474.5]}, "98": {"corners": [[968.0, 338.0], [1074.0, 323.0], [1091.0, 415.0], [982.0, 429.0]], "orientation": [-0.16701945662498474, -0.9859535694122314], "center": [1028.75, 376.25]}, "45": {"corners": [[1154.0, 402.0], [1135.0, 313.0], [1241.0, 299.0], [1263.0, 388.0]], "orientation": [-0.9916261434555054, 0.12914201617240906], "center": [1198.25, 350.5]}, "38": {"corners": [[1572.0, 294.0], [1600.0, 386.0], [1488.0, 394.0], [1464.0, 303.0]], "orientation": [0.9970277547836304, -0.07704305648803711], "center": [1531.0, 344.25]}, "47": {"corners": [[557.0, 279.0], [655.0, 286.0], [632.0, 374.0], [530.0, 366.0]], "orientation": [0.2747211158275604, -0.9615239500999451], "center": [593.5, 326.25]}, "21": {"corners": [[663.0, 226.0], [563.0, 232.0], [571.0, 153.0], [668.0, 147.0]], "orientation": [-0.08200138807296753, 0.9966322183609009], "center": [616.25, 189.5]}, "89": {"corners": [[1409.0, 148.0], [1510.0, 142.0], [1532.0, 223.0], [1427.0, 230.0]], "orientation": [-0.23832756280899048, -0.9711848497390747], "center": [1469.5, 185.75]}, "35": {"corners": [[879.0, 46.0], [880.0, 118.0], [782.0, 123.0], [785.0, 51.0]], "orientation": [0.9986464381217957, -0.05201283469796181], "center": [831.5, 84.5]}, "31": {"corners": [[702.0, 103.0], [609.0, 109.0], [616.0, 37.0], [707.0, 31.0]], "orientation": [-0.08304548263549805, 0.9965457916259766], "center": [658.5, 70.0]}, "97": {"corners": [[1013.0, 30.0], [1034.0, 100.0], [939.0, 119.0], [920.0, 48.0]], "orientation": [0.9811782240867615, -0.19310422241687775], "center": [976.5, 74.25]}, "1": {"corners": [[1369.0, 28.0], [1464.0, 21.0], [1486.0, 94.0], [1388.0, 102.0]], "orientation": [-0.2686575949192047, -0.9632357954978943], "center": [1426.75, 61.25]}, "33": {"corners": [[1185.0, 89.0], [1088.0, 96.0], [1080.0, 23.0], [1174.0, 16.0]], "orientation": [0.12904880940914154, 0.9916382431983948], "center": [1131.75, 56.0]}, "95": {"corners": [[1213.0, 12.0], [1308.0, 6.0], [1324.0, 77.0], [1226.0, 85.0]], "orientation": [-0.19742515683174133, -0.980318009853363], "center": [1267.75, 45.0]}}

# fancy maze. target is 38.
#  0 1 2 3 4 5 6 7 8 9101112131415161718192021222324
# 0 - - - - - * - - - - - - - - - - - - - - - - - - -
# 1 - - * * * * * * * * * * * * * * * * * * * * * * *
# 2 - - * * - - - - - - - - - - - * * * - - - * * * -
# 3 - - * * - - - - - - - - - - S * * E - - - - * * -
# 4 - * * * * * * - - * * - - - - * * - - - - - * * -
# 5 - * * * * * * - - * * * * * * * * * * * * - * * -
# 6 - * * - - - - - - * * - - - - - * * * * * - * * -
# 7 - * * - - - - - - * * - - - - - - - - - - - * * -
# 8 - * * - - - - - - * - - * * - - - - - - - - * * -
# 9 - * - - - - - - - - - - * * - - - - - - - - * * -
#10 - * * * * * * * * - - - * * - - - - - - - - * * -
#11 - * * * * * * * - * * * * * * * * * * * * * * * -
example7 = {"time": 7869.709783838, "41": {"corners": [[923.0, 739.0], [988.0, 755.0], [970.0, 822.0], [906.0, 805.0]], "orientation": [0.2544932961463928, -0.9670745134353638], "center": [946.75, 780.25]}, "89": {"corners": [[1167.0, 373.0], [1163.0, 431.0], [1100.0, 425.0], [1105.0, 368.0]], "orientation": [0.9961503148078918, 0.08766122907400131], "center": [1133.75, 399.25]}, "62": {"corners": [[758.0, 413.0], [698.0, 405.0], [712.0, 349.0], [771.0, 357.0]], "orientation": [-0.23435769975185394, 0.972150444984436], "center": [734.75, 381.0]}, "robot": {"corners": [[989.0, 260.0], [997.0, 203.0], [1061.0, 208.0], [1054.0, 266.0]], "orientation": [-0.9963841438293457, -0.08496298640966415], "center": [1025.25, 234.25]}, "21": {"corners": [[1208.0, 104.0], [1208.0, 155.0], [1148.0, 155.0], [1149.0, 104.0]], "orientation": [1.0, 0.0], "center": [1178.25, 129.5]}, "34": {"corners": [[1666.0, 376.0], [1730.0, 381.0], [1733.0, 440.0], [1668.0, 434.0]], "orientation": [-0.04269607365131378, -0.9990881085395813], "center": [1699.25, 407.75]}, "40": {"corners": [[1352.0, 90.0], [1411.0, 87.0], [1419.0, 136.0], [1359.0, 141.0]], "orientation": [-0.14834044873714447, -0.9889363646507263], "center": [1385.25, 113.5]}, "56": {"corners": [[644.0, 137.0], [589.0, 131.0], [602.0, 82.0], [657.0, 87.0]], "orientation": [-0.2540123760700226, 0.9672009944915771], "center": [623.0, 109.25]}, "58": {"corners": [[463.0, 129.0], [408.0, 134.0], [410.0, 85.0], [466.0, 80.0]], "orientation": [-0.05095412954688072, 0.9987009763717651], "center": [436.75, 107.0]}, "48": {"corners": [[851.0, 920.0], [855.0, 849.0], [924.0, 850.0], [922.0, 922.0]], "orientation": [-0.9997705221176147, -0.02142365463078022], "center": [888.0, 885.25]}, "46": {"corners": [[1025.0, 924.0], [955.0, 918.0], [961.0, 846.0], [1030.0, 852.0]], "orientation": [-0.07616698741912842, 0.9970951080322266], "center": [992.75, 885.0]}, "25": {"corners": [[1586.0, 843.0], [1599.0, 915.0], [1527.0, 919.0], [1515.0, 847.0]], "orientation": [0.9984387755393982, -0.05585671588778496], "center": [1556.75, 881.0]}, "95": {"corners": [[1657.0, 914.0], [1646.0, 842.0], [1717.0, 841.0], [1729.0, 912.0]], "orientation": [-0.999779999256134, 0.020974406972527504], "center": [1687.25, 877.25]}, "26": {"corners": [[1388.0, 831.0], [1459.0, 834.0], [1463.0, 907.0], [1391.0, 904.0]], "orientation": [-0.04789019376039505, -0.9988526105880737], "center": [1425.25, 869.0]}, "42": {"corners": [[1246.0, 899.0], [1177.0, 906.0], [1167.0, 834.0], [1235.0, 827.0]], "orientation": [0.14430689811706543, 0.989533007144928], "center": [1206.25, 866.5]}, "44": {"corners": [[1135.0, 902.0], [1066.0, 896.0], [1070.0, 824.0], [1139.0, 830.0]], "orientation": [-0.055470023304224014, 0.9984604120254517], "center": [1102.5, 863.0]}, "24": {"corners": [[1341.0, 823.0], [1350.0, 895.0], [1278.0, 897.0], [1272.0, 826.0]], "orientation": [0.9993718862533569, -0.035438720136880875], "center": [1310.25, 860.25]}, "49": {"corners": [[804.0, 886.0], [735.0, 887.0], [738.0, 816.0], [805.0, 815.0]], "orientation": [-0.028157847002148628, 0.9996035099029541], "center": [770.5, 851.0]}, "77": {"corners": [[581.0, 813.0], [646.0, 795.0], [659.0, 862.0], [593.0, 880.0]], "orientation": [-0.18340258300304413, -0.9830378293991089], "center": [619.75, 837.5]}, "74": {"corners": [[480.0, 779.0], [544.0, 790.0], [526.0, 858.0], [461.0, 847.0]], "orientation": [0.2625170052051544, -0.9649273753166199], "center": [502.75, 818.5]}, "75": {"corners": [[350.0, 774.0], [414.0, 779.0], [400.0, 847.0], [336.0, 842.0]], "orientation": [0.20165292918682098, -0.9794570803642273], "center": [375.0, 810.5]}, "72": {"corners": [[249.0, 771.0], [312.0, 776.0], [297.0, 843.0], [234.0, 838.0]], "orientation": [0.21847234666347504, -0.9758431315422058], "center": [273.0, 807.0]}, "1": {"corners": [[170.0, 836.0], [107.0, 837.0], [118.0, 770.0], [180.0, 769.0]], "orientation": [-0.15482667088508606, 0.9879416227340698], "center": [143.75, 803.0]}, "92": {"corners": [[1723.0, 798.0], [1653.0, 790.0], [1651.0, 722.0], [1720.0, 729.0]], "orientation": [0.03647206723690033, 0.9993346333503723], "center": [1686.75, 759.75]}, "73": {"corners": [[96.0, 626.0], [155.0, 627.0], [143.0, 690.0], [83.0, 689.0]], "orientation": [0.1946188360452652, -0.9808788895606995], "center": [119.25, 658.0]}, "51": {"corners": [[983.0, 629.0], [978.0, 693.0], [911.0, 688.0], [917.0, 625.0]], "orientation": [0.9977182149887085, 0.06751476973295212], "center": [947.25, 658.75]}, "94": {"corners": [[1716.0, 687.0], [1648.0, 676.0], [1651.0, 611.0], [1719.0, 622.0]], "orientation": [-0.04610477015376091, 0.998936653137207], "center": [1683.5, 649.0]}, "79": {"corners": [[667.0, 589.0], [679.0, 528.0], [741.0, 535.0], [729.0, 596.0]], "orientation": [-0.9936867356300354, -0.11219044029712677], "center": [704.0, 562.0]}, "70": {"corners": [[133.0, 521.0], [187.0, 533.0], [163.0, 593.0], [108.0, 580.0]], "orientation": [0.38074979186058044, -0.9246780872344971], "center": [147.75, 556.75]}, "30": {"corners": [[1650.0, 508.0], [1715.0, 510.0], [1723.0, 572.0], [1656.0, 570.0]], "orientation": [-0.11219044029712677, -0.9936867356300354], "center": [1686.0, 540.0]}, "71": {"corners": [[124.0, 456.0], [183.0, 449.0], [179.0, 507.0], [120.0, 507.0]], "orientation": [0.0731976106762886, -0.9973174929618835], "center": [151.5, 479.75]}, "76": {"corners": [[704.0, 437.0], [764.0, 447.0], [748.0, 505.0], [688.0, 495.0]], "orientation": [0.2659290134906769, -0.9639926552772522], "center": [726.0, 471.0]}, "90": {"corners": [[1370.0, 476.0], [1307.0, 459.0], [1323.0, 403.0], [1386.0, 421.0]], "orientation": [-0.27700695395469666, 0.9608678817749023], "center": [1346.5, 439.75]}, "96": {"corners": [[1421.0, 465.0], [1412.0, 405.0], [1476.0, 402.0], [1485.0, 461.0]], "orientation": [-0.9985079169273376, 0.05460590124130249], "center": [1448.5, 433.25]}, "27": {"corners": [[1196.0, 385.0], [1259.0, 390.0], [1256.0, 449.0], [1193.0, 444.0]], "orientation": [0.05078185349702835, -0.9987097978591919], "center": [1226.0, 417.0]}, "50": {"corners": [[1062.0, 377.0], [1061.0, 434.0], [998.0, 433.0], [999.0, 376.0]], "orientation": [0.9998740553855896, 0.015871016308665276], "center": [1030.0, 405.0]}, "60": {"corners": [[898.0, 429.0], [907.0, 372.0], [968.0, 379.0], [959.0, 436.0]], "orientation": [-0.9934800863265991, -0.114005908370018], "center": [933.0, 404.0]}, "63": {"corners": [[803.0, 372.0], [864.0, 374.0], [859.0, 431.0], [797.0, 429.0]], "orientation": [0.09604514390230179, -0.9953770041465759], "center": [830.75, 401.5]}, "66": {"corners": [[284.0, 414.0], [226.0, 415.0], [235.0, 360.0], [293.0, 358.0]], "orientation": [-0.16007116436958313, 0.9871054887771606], "center": [259.5, 386.75]}, "64": {"corners": [[387.0, 412.0], [327.0, 421.0], [327.0, 364.0], [386.0, 356.0]], "orientation": [0.008849211037158966, 0.9999608397483826], "center": [356.75, 388.25]}, "67": {"corners": [[190.0, 410.0], [134.0, 410.0], [146.0, 354.0], [202.0, 354.0]], "orientation": [-0.2095290869474411, 0.9778023958206177], "center": [168.0, 382.0]}, "65": {"corners": [[491.0, 404.0], [430.0, 419.0], [422.0, 365.0], [482.0, 350.0]], "orientation": [0.1554928719997406, 0.9878370761871338], "center": [456.25, 384.5]}, "53": {"corners": [[1175.0, 345.0], [1113.0, 342.0], [1117.0, 285.0], [1178.0, 290.0]], "orientation": [-0.06237828731536865, 0.9980525970458984], "center": [1145.75, 315.5]}, "20": {"corners": [[1670.0, 285.0], [1733.0, 291.0], [1734.0, 347.0], [1671.0, 340.0]], "orientation": [-0.018015094101428986, -0.9998377561569214], "center": [1702.0, 315.75]}, "68": {"corners": [[218.0, 323.0], [162.0, 322.0], [174.0, 269.0], [229.0, 268.0]], "orientation": [-0.20829197764396667, 0.9780666828155518], "center": [195.75, 295.5]}, "38": {"corners": [[1251.0, 290.0], [1260.0, 236.0], [1320.0, 244.0], [1313.0, 299.0]], "orientation": [-0.9904307126998901, -0.13801082968711853], "center": [1286.0, 267.25]}, "43": {"corners": [[1124.0, 200.0], [1183.0, 200.0], [1185.0, 253.0], [1125.0, 253.0]], "orientation": [-0.028290558606386185, -0.9995997548103333], "center": [1154.25, 226.5]}, "22": {"corners": [[1637.0, 242.0], [1626.0, 190.0], [1686.0, 186.0], [1699.0, 238.0]], "orientation": [-0.9978569149971008, 0.06543324142694473], "center": [1662.0, 214.0]}, "69": {"corners": [[245.0, 170.0], [240.0, 219.0], [183.0, 225.0], [189.0, 175.0]], "orientation": [0.9952954053878784, -0.09688716381788254], "center": [214.25, 197.25]}, "28": {"corners": [[1555.0, 120.0], [1613.0, 116.0], [1625.0, 167.0], [1565.0, 171.0]], "orientation": [-0.21083787083625793, -0.9775210022926331], "center": [1589.5, 143.5]}, "55": {"corners": [[927.0, 147.0], [869.0, 144.0], [875.0, 94.0], [933.0, 96.0]], "orientation": [-0.11798206716775894, 0.993015706539154], "center": [901.0, 120.25]}, "57": {"corners": [[830.0, 145.0], [773.0, 145.0], [777.0, 94.0], [834.0, 94.0]], "orientation": [-0.07819124311208725, 0.9969383478164673], "center": [803.5, 119.5]}, "54": {"corners": [[741.0, 145.0], [684.0, 143.0], [690.0, 93.0], [747.0, 95.0]], "orientation": [-0.1191452145576477, 0.9928768277168274], "center": [715.5, 119.0]}, "52": {"corners": [[1106.0, 142.0], [1047.0, 142.0], [1049.0, 92.0], [1107.0, 93.0]], "orientation": [-0.03028912842273712, 0.9995412230491638], "center": [1077.25, 117.25]}, "32": {"corners": [[1301.0, 149.0], [1242.0, 139.0], [1254.0, 89.0], [1311.0, 99.0]], "orientation": [-0.21486179530620575, 0.9766445159912109], "center": [1277.0, 119.0]}, "88": {"corners": [[959.0, 140.0], [958.0, 89.0], [1015.0, 87.0], [1017.0, 137.0]], "orientation": [-0.9990561604499817, 0.04343722388148308], "center": [987.25, 113.25]}, "93": {"corners": [[264.0, 133.0], [209.0, 136.0], [216.0, 88.0], [270.0, 84.0]], "orientation": [-0.13283298909664154, 0.9911384582519531], "center": [239.75, 110.25]}, "59": {"corners": [[554.0, 131.0], [497.0, 135.0], [499.0, 86.0], [556.0, 83.0]], "orientation": [-0.04120209813117981, 0.9991508722305298], "center": [526.5, 108.75]}, "33": {"corners": [[1712.0, 131.0], [1701.0, 82.0], [1760.0, 79.0], [1772.0, 129.0]], "orientation": [-0.999118447303772, 0.04197976738214493], "center": [1736.25, 105.25]}, "36": {"corners": [[1519.0, 129.0], [1460.0, 131.0], [1454.0, 80.0], [1513.0, 80.0]], "orientation": [0.1191452145576477, 0.9928768277168274], "center": [1486.5, 105.0]}, "61": {"corners": [[378.0, 116.0], [322.0, 124.0], [322.0, 77.0], [378.0, 69.0]], "orientation": [0.0, 1.0], "center": [350.0, 96.5]}}


# NOTE: coordinates are in (x, y) format, e.g. S is at (0, 4).
#   0 1 2 3 4 5 6 7 8 9
# 0
# 1                    
# 2       * *          
# 3         * *        
# 4 S         *       E
# 5           *
# 6         * *
# 7       * *
# 8
# 9
grid_example1 = {"grid": {(3,2), (4,2), (4,3), (5,3), (5,4),
                          (5,5), (5,6), (4,6), (4,7), (3,7)},
        "start": (0,4), "end": (9,4)}

# The minimum speed for an individual wheel.
min_speed = 3

# The maximum proportion of the outer wheel speed to the inner wheel
# speed when turning.
max_proportion = 3.5

def grid_coordinates(cam_coordinates, cell_length):
    return tuple(int(x / cell_length) for x in cam_coordinates)

def cam_coordinates(grid_coordinates, cell_length):
    return tuple(int(x * cell_length + cell_length/2)
                 for x in grid_coordinates)

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    # always returns a nonnegative angle.
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def radius(square):
    return np.linalg.norm(np.subtract(square["center"],
        square["corners"][0])) * 2

def attraction_field(robot, target):
    vector = np.subtract(target, robot["center"])
    distance = np.linalg.norm(vector)
    r = radius(robot)
    if distance < r:
        return (0, 0)
    else:
        return vector

def closer_side(robot, target, front_side="front"):
    offset = 0 if front_side == "front" else 2
    distance = lambda corner: np.linalg.norm(np.subtract(
        target["center"], corner))
    return "left" if distance(robot["corners"][0+offset]) < distance(
            robot["corners"][1+offset]) else "right"

def get_command(robot, vector):
    angle = angle_between(robot["orientation"], vector)
    front_side = "front"
    if angle > math.pi/2:
        front_side = "back"
        angle = math.pi - angle
    target_wrapper = {"center": np.add(robot["center"],
                                       np.multiply(vector, 50))}
    turn_direction = closer_side(robot, target_wrapper, front_side)

    if debug:
        print("vector:", vector)
        print("orientation:", robot["orientation"])
        print("angle:", angle)

    proportion = 1 + (max_proportion - 1) * angle / (math.pi/2)

    inner_wheel_speed = min_speed
    outer_wheel_speed = min_speed * proportion
    command = [inner_wheel_speed, outer_wheel_speed]

    if front_side == "back":
        command.reverse()
        command = [x * -1 for x in command]

    if turn_direction == "right":
        command.reverse()

    return command

def positions(data, target_num):
    robot = data.get("robot", None)
    target = data.get(target_num, None)
    obstacles = tuple(data[key] for key in data
                      if key not in ('time', 'robot', target_num))
    return robot, target, obstacles

def get_grid(obstacles):
    cell_length = max(np.linalg.norm(np.subtract(*o["corners"][:2]))
                      for o in obstacles)
    corners = tuple(corner for o in obstacles for corner in o["corners"])
    occupied = {grid_coordinates(corner, cell_length) for corner in corners}
    return occupied, cell_length

def get_path(robot, target, obstacles, algorithm="astar"):
    # Get the path to follow.
    # grid will be a set of coordinates for cells that don't have an obstacle.
    # cell_length is the pixel width of each cell in the grid.
    # path will be a list of grid coordinates.
    grid, cell_length = get_grid(obstacles)
    path = (astar if algorithm == "astar" else rrt)(grid,
            grid_coordinates(robot["center"], cell_length),
            grid_coordinates(target["center"], cell_length))
    return path, cell_length

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from.keys():
        current = came_from[current]
        total_path.append(current)
    total_path.reverse()
    return total_path

def adjacent_cells(occupied, cell):
    directions = ((x, y) for x in (-1, 0, 1)
                         for y in (-1, 0, 1)
                         if (x, y) != (0, 0))
    neighbors = {tuple(np.add(cell, d)) for d in directions}
    return set.difference(neighbors, occupied)

def astar(grid, start, end):
    dist = lambda a, b: np.linalg.norm(np.subtract(b, a))
    closed_set = set()
    open_set = {start}
    came_from = {}
    gscore = {start: 0}
    fscore = {start: dist(start, end)}

    while open_set:
        current = min(open_set, key=lambda cell: fscore.get(cell, float('inf')))
        if current == end:
            return reconstruct_path(came_from, current)[1:]

        open_set.remove(current)
        closed_set.add(current)
        for neighbor in adjacent_cells(grid, current):
            if neighbor in closed_set:
                continue		
            tentative_gscore = (gscore.get(current, float('inf')) +
                                dist(current, neighbor))
            if neighbor not in open_set:
                open_set.add(neighbor)
            elif tentative_gscore >= gscore.get(neighbor, float('inf')):
                continue

            came_from[neighbor] = current
            gscore[neighbor] = tentative_gscore
            fscore[neighbor] = gscore[neighbor] + dist(neighbor, end)

    raise Exception('no path found')

def rrt(grid, start, end):
    # TODO implement
    path = []
    return path

def print_grid(robot, target, obstacles):
    grid, cell_length = get_grid(obstacles)
    start = grid_coordinates(robot["center"], cell_length)
    end = grid_coordinates(target["center"], cell_length)
    squares = set.union(grid, {start, end})

    def symbol(coords):
        if coords == start:
            return 'S'
        elif coords in grid:
            return '*'
        elif coords == end:
            return 'E'
        else:
            return '-'

    print(repr(grid))
    max_x, max_y = [max(coords[i] for coords in squares) for i in (0,1)]
    print(" " + "".join("{:2d}".format(x) for x in range(max_x + 1)))
    for y in range(max_y):
        print("{:2d}".format(y), *[symbol((x, y)) for x in range(max_x + 1)])

def main(host, port, target_num, algorithm="astar"):
    loop = asyncio.get_event_loop()
    reader, writer = loop.run_until_complete(
        asyncio.open_connection(host, port))
    print(reader.readline())

    def do(command):
        print('>>>', command)
        writer.write(command.strip().encode())
        res = loop.run_until_complete(reader.readline()).decode().strip()
        print('<<<', res)
        print()
        return res

    def get_positions():
        while True:
            try:
                data = json.loads(do('where'))
                if "robot" in data:
                    return positions(data, target_num)
            except json.decoder.JSONDecodeError:
                pass
            print("server returned bad response")
            sleep(0.1)

    robot, target, obstacles = get_positions()
    if target == None:
        raise Exception("Can't see target (#{})".format(target_num))
    print_grid(robot, target, obstacles)
    input("Press Enter to start")
    path, cell_length = get_path(robot, target, obstacles, algorithm)
    while path:
        target = cam_coordinates(path[0], cell_length)
        vector = attraction_field(robot, target)
        if not any(vector):
            path.pop(0)
            continue

        arg_list = map(lambda x: int(round(x)), get_command(robot, vector))

        if debug:
            print("command:", list(arg_list))
            print()
            input("press Enter")
        else:
            do("speed " + " ".join(str(arg) for arg in arg_list))
            sleep(0.1)
        robot, _, _ = get_positions()

    do("power 0 0")
    writer.close()

def run_tests():
    args = tuple(grid_example1[k] for k in ["grid", "start", "end"])
    print("a* with grid_example1:", astar(*args))
    print("rrt with grid_example1:", rrt(*args))
    print()

    robot, target, obstacles = positions(example5, "25")
    print("grid with example5:", get_grid(obstacles))
    print("a* with example5:", get_path(robot, target, obstacles, "astar")[0])
    print("rrt with example5:", get_path(robot, target, obstacles, "rrt")[0])
    print()

    robot, target, obstacles = positions(example6, "29")
    print_grid(robot, target, obstacles)
    print(get_path(robot, target, obstacles, "astar")[0])
    
if __name__ == '__main__':
    from sys import argv
    run_tests()
    #main(*argv[1:])
