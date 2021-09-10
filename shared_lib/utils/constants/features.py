# SRC alone
SRC_AD1 = 'src_ad1'

# TGT alone
VERD_AD1 = 'verd_ad1'
PC_AD1 = 'pc_ad1'

# SRC vs VERD
SRC_VERD_AR = 'src_verd_ar'
SRC_VERD_AP = 'src_verd_ap'
SRC_VERD_R1_R = 'src_verd_r1_r'
SRC_VERD_R1_P = 'src_verd_r1_p'
SRC_VERD_R2_R = 'src_verd_r2_r'
SRC_VERD_R2_P = 'src_verd_r2_p'
SRC_VERD_LEN_DIFF = 'src_verd_len_diff'

# SRC vs PROs & CONs
SRC_PC_AR = 'src_pc_ar'
SRC_PC_AP = 'src_pc_ap'
SRC_PC_R1_R = 'src_pc_r1_r'
SRC_PC_R1_P = 'src_pc_r1_p'
SRC_PC_R2_R = 'src_pc_r2_r'
SRC_PC_R2_P = 'src_pc_r2_p'
SRC_PC_LEN_DIFF = 'src_pc_len_diff'

# SRC vs REST
REST_AR = 'rest_ar'
REST_AP = 'rest_ap'
REST_R1_R = 'rest_r1_r'
REST_R1_P = 'rest_r1_p'
REST_R2_R = 'rest_r2_r'
REST_R2_P = 'rest_r2_p'

FEATURE_ORDER = [
    SRC_AD1, VERD_AD1, PC_AD1,
    
    SRC_VERD_AR, SRC_VERD_AP, SRC_VERD_R1_R, SRC_VERD_R1_P, SRC_VERD_R2_R, SRC_VERD_R2_P, SRC_VERD_LEN_DIFF,
    SRC_PC_AR, SRC_PC_AP, SRC_PC_R1_R, SRC_PC_R1_P, SRC_PC_R2_R, SRC_PC_R2_P, SRC_PC_LEN_DIFF,
    
    REST_AP, REST_AR, REST_R1_P, REST_R1_R, REST_R2_P, REST_R2_R
]
