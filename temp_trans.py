"""
 Temp transform
"""
tr = 'bc'
df_tmp = df_org.copy()
tmp = df_tmp.columns
for c in tmp:
    s = df_tmp[c]
    print("Transforming {} by application of {}:".format(c, tr))
    if tr == 'sqrt':
        s_trans = np.sqrt(s)
    elif tr == 'quart':
        s_trans = np.power(s, .25)
    elif tr == 'log':
        s_trans = np.log(s+1)
    elif tr == 'log2':
        s_trans = np.log2(s+1)
    elif tr == 'log10':
        s_trans = np.log10(s+1)
    elif tr == 'log1p':
        s_trans = np.log1p(s)
    elif tr == 'bc':
        s_trans = boxcox(s+1)[0]


    s_stats = describe(s_trans)
    df_tmp[c] = s_trans
    print("{}\n".format(s_stats))

df_tmp.hist()
