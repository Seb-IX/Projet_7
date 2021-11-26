import re
def clean_url(sentence):
    sentence = re.sub('http[s]*://[0-9a-zA-Z-_.]*.[a-z]{0,3}/[0-9a-zA-Z-_./]*', '', sentence)
    return sentence

def clean_smile(sentence):
    sentence = re.sub(';\)','emote_clin_doeil ',sentence)
    sentence = re.sub(':\)|=\)','emote_sourire ',sentence)
    sentence = re.sub('x\)','emote_cross_smile ',sentence)
    sentence = re.sub('x{1,}[Dd]{1,}[^A-Za-z0-9]','emote_mdr ',sentence)
    sentence = re.sub(';[pP]{1,}[^A-Za-z0-9]|:[pP]{1,}[^A-Za-z0-9]|x[pP]{1,}[^A-Za-z0-9]|=[pP]{1,}[^A-Za-z0-9]',
                      'emote_tire_langue ',sentence)
    sentence = re.sub('[oO]_[oO]','emote_tres_surpris',sentence)
    sentence = re.sub('[Xx]{1,}[Oo]{1,}[^A-Za-z0-9]','emote_ko ',sentence)
    sentence = re.sub('[bB]\)[^A-Za-z0-9]','emote_lunette ',sentence)
    sentence = re.sub('[=:;][oO]{1,}[^A-Za-z0-9]','emote_surpris ',sentence)
    sentence = re.sub('[=:;][S]{1,}[^A-Za-z0-9]','emote_genant ',sentence)
    sentence = re.sub('[:;=][dD]{1,}[^A-Za-z0-9]','emote_gros_sourire ',sentence)
    sentence = re.sub('[:;=][$]{1,}[^A-Za-z0-9]','emote_genant_discret ',sentence)
    sentence = re.sub('[:;=]\|[^A-Za-z0-9]','emote_neutre ',sentence)
    sentence = re.sub('[:;=]/[^A-Za-z0-9]','emote_deception ',sentence)
    sentence = re.sub('[:;=]\([^A-Za-z0-9]','emote_insatisfait ',sentence)
    sentence = re.sub('\*[-_]\*[^A-Za-z0-9]','emote_magnifique ',sentence)
    sentence = re.sub('\^[-_]{0,1}\^[^A-Za-z0-9]','emote_joyeux ',sentence)
    sentence = re.sub('--\'[^A-Za-z0-9]','emote_enerve ',sentence)
    return sentence

def clean_escape(sentence):
    sentence = re.sub('&amp;','&',sentence)
    sentence = re.sub('&quot;','\"',sentence)
    sentence = re.sub('&gt;','>',sentence)
    sentence = re.sub('&lt;','<',sentence)
    return sentence

def clean_uniuque_char(sentence):
    sentence = " ".join([v for v in sentence.split(" ") if ((len(v) > 1) or (not v.isalpha()))])
    return sentence

def cleanning_data(sentence):
    sentence += " "
    sentence = re.sub('\n+', ' ', sentence.lower())
    # special caractère
    sentence = re.sub('[éèêë]','e',sentence)
    sentence = re.sub('[ïîì]','i',sentence)
    sentence = re.sub('[àâä]','a',sentence)
    sentence = re.sub('[ôö]','o',sentence) 
    sentence = re.sub('[üûù]','u',sentence)
    sentence = re.sub('[ç]','c',sentence)
    # personal_clean
    sentence = clean_escape(sentence) # usless ?
    sentence = clean_url(sentence)
    sentence = clean_smile(sentence)
    sentence = re.sub(r'[^a-z@#\'_ ]+',' ',sentence)
    sentence = clean_uniuque_char(sentence)
    sentence = re.sub(' {2,}', ' ', sentence)
    return sentence