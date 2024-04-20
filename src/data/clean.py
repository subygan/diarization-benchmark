import os
dir_path = os.path.dirname(os.path.realpath(__file__))

wav_dir = os.path.join(dir_path,"test_wav")
rttm_dir = os.path.join(dir_path,"test_rttm")

wav_list = os.listdir(wav_dir)
wav_list.sort()
rttm_list = os.listdir(rttm_dir)
rttm_list.sort()


# cleanup only english audio for now

en_audio_files = {
            "aepyx", "aggyz", "aiqwk", "aorju", "auzru", "bjruf", "bmsyn", "bvqnu",
            "bvyvm", "bxcfq", "cadba", "cawnd", "clfcg", "cpebh", "cqfmj", "crorm",
            "crylr", "cvofp", "dgvwu", "dkabn", "dlast", "dohag", "dxbbt", "dxokr",
            "dzsef", "dzxut", "eazeq", "eddje", "eguui", "eoyaz", "epygx", "erslt",
            "eucfa", "euqef", "ezxso", "fpfvy", "fqrnu", "fvhrk", "fxnwf", "fyqoe",
            "fzwtp", "gcfwp", "gfneh", "gkiki", "gmmwm", "gtjow", "gtnjb", "gukoa",
            "gwloo", "gylzn", "gyomp", "hcyak", "heolf", "hhepf", "hqhrb", "iabca",
            "iacod", "ibrnm", "ifwki", "iiprr", "iowob", "isrps", "isxwc", "jbowg",
            "jdrwl", "jeymh", "jgiyq", "jjvkx", "jrfaz", "jttar", "jwggf", "jxpom",
            "jxydp", "kajfh", "kgjaa", "kmjvh", "kmunk", "kpjud", "kvkje", "kzmyi",
            "laoyl", "lbfnx", "ledhe", "leneg", "lhuly", "lilfy", "ljpes", "lkikz",
            "lpola", "ltgmz", "lubpm", "luobn", "mbzht", "mclsr", "mjmgr", "mkhie",
            "mqtep", "msbyq", "mupzb", "mxdpo", "mxduo", "myjoe", "neiye", "nitgx",
            "nlvdr", "nprxc", "nqcpi", "nqyqm", "ocfop", "ofbxh", "olzkb", "ooxlj",
            "oqwpd", "otmpf", "ouvtt", "pccww", "pgtkk", "pkwrt", "poucc", "ppexo",
            "pxqme", "pzxit", "qadia", "qajyo", "qeejz", "qlrry", "qoarn", "qwepo",
            "qxana", "ralnu", "rarij", "rmvsh", "rpkso", "rsypp", "rxulz", "ryken",
            "sbrmv", "sebyw", "sfdvy", "svxzm", "swbnm", "sxqvt", "thnuq", "tiido",
            "tkhgs", "tkybe", "tnjoh", "tpnyf", "tpslg", "tvtoe", "uedkc", "uevxo",
            "uicid", "upshw", "uqxlg", "usqam", "vdlvr", "vgaez", "vgevv", "vncid",
            "vtzqw", "vuewy", "vzuru", "wcxfk", "wdvva", "wemos", "wibky", "wlfsf",
            "wprog", "wwvcs", "wwzsk", "xggbk", "xkmqx", "xlsme", "xlyov", "xmyyy",
            "xqxkt", "xtdcl", "xtzoq", "xvxwv", "ybhwz", "ygrip", "ylgug", "ytmef",
            "ytula", "yukhy", "zedtj", "zehzu", "zowse", "zqidv", "zsgto", "zzsba",
            "zztbo",
        }


for i,j in zip(wav_list,rttm_list):
    if i.split(".")[0] in en_audio_files:
        continue
    else:
        os.remove(os.path.join(wav_dir,i))
        os.remove(os.path.join(rttm_dir,j))

print(len(en_audio_files))
print(len(wav_list))