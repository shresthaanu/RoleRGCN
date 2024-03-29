import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def impute_values(df, features, impute):
    imputed_df = df
    if impute:
        imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
        for col in features:
          try:
              imputed_column = imputer.fit_transform(df[col].values.reshape(-1, 1))
              #Fill in Series on DataFrame
              imputed_df[col] = imputed_column
          except:
              pass
    else:
        imputed_df = df.dropna(subset=features)
    return imputed_df

def get_features(df, node_type, impute=True, scale=True):
  if node_type == 'user':
      personality_feature_all = ['facet_cheerfulness_raw','facet_gregariousness_raw','value_self_transcendence_raw','value_self_enhancement_raw',
                                'value_openness_to_change_raw','value_conservation_raw','need_excitement_raw','facet_modesty_raw','facet_morality_raw','facet_sympathy_raw',
                                'facet_trust_raw','need_challenge_raw','need_love_raw','facet_orderliness_raw','facet_self_discipline_raw','facet_liberalism_raw','facet_activity_level_raw',
                                'facet_emotionality_raw','facet_excitement_seeking_raw','facet_altruism_raw','big5_neuroticism_raw','facet_immoderation_raw','need_closeness_raw',
                                'value_hedonism_raw','facet_imagination_raw','facet_assertiveness_raw','big5_conscientiousness_raw','facet_anger_raw','facet_dutifulness_raw',
                                'facet_achievement_striving_raw','facet_adventurousness_raw','facet_vulnerability_raw','facet_cautiousness_raw','need_ideal_raw','big5_agreeableness_raw','need_practicality_raw',
                                'facet_depression_raw','big5_extraversion_raw','big5_openness_raw','facet_intellect_raw','need_stability_raw','need_harmony_raw','need_structure_raw',
                                'facet_self_consciousness_raw','facet_artistic_interests_raw','facet_anxiety_raw','facet_self_efficacy_raw','need_curiosity_raw','need_liberty_raw','facet_friendliness_raw']

      personality_feature_big5 = [x for x in personality_feature_all if x.startswith('big5')]
      personality_feature_need = [x for x in personality_feature_all if x.startswith('need')]
      personality_feature_facet = [x for x in personality_feature_all if x.startswith('facet')]
      personality_feature_value = [x for x in personality_feature_all if x.startswith('value')]
  #     print(personality_feature_need)
      
      tff_feature = ['TFF']
      
      demographics_feature = [x for x in df.columns if x.startswith('age')]+\
      [x for x in df.columns if x.startswith('gender')]+[x for x in df.columns if x.startswith('org')]
  #     print(demographics_feature)
      
      stress_feature = ['stress']
      
      polar = ['Chi Score']
      
      behavior = ['count_day_posts','per_posts_time','time_diff_avg','count_night_posts','per_posts_day']
      
      emotion = ['Anger','Disgust','sad','happy','dont_care','annoyed','afraid','Fear','Sadness',
                'Surprise','Trust', 'Anticipation','angry','inspired','amused','Joy']
      
      readability = ['gunning_fog_index','flesch_kincaid_grade_level', 'lix_index','flesch_reading_ease','dale_chall_known_fraction',
                    'smog_index','ari_index','dale_chall_score','coleman_liau_index']
                    
      stylistic = ['count_hashtags', 'count_urls', 'count_RT','count_user', 'count_words', 'count_uppercased', 'count_lowercased','count_punctuation',
                  'count_breaking','count_emoji','count_trailing_period', 'count_stopwords', 'count_char','count_uppercased_char', 'count_lowercased_char']

      selected_feature = personality_feature_all+demographics_feature+tff_feature+behavior+readability+emotion+stylistic
  elif node_type == 'news':
      news_features = ['Analytic','insight','cause','discrep','tentat','certain','differ','affiliation','power','reward','risk','work','leisure',
                            'money','relig','Tone','affect','WC','WPS','num_nouns','num_propernouns','num_personalnouns','num_ppssessivenouns',
                            'num_whpronoun','num_determinants','num_whdeterminants','num_cnum','num_adverb','num_interjections','num_verb','num_adj',
                            'num_vbd','num_vbg','num_vbn','num_vbp','num_vbz','focuspast','focusfuture','i','we','you','shehe','quant','compare','Exclam',
                            'negate','swear','netspeak','interrog','count_uppercased','percentage_stopwords','AllPunc','Quote', 'lexical_diversity','wlen',
                            'gunning_fog_index','smog_index','flesch_kincaid_grade_level','Anger','Anticipation','Disgust','Fear','Joy', 
                                                    'Sadness', 'Surprise', 'Trust','neg','pos','posemo','negemo','anx']
      selected_feature = news_features
  elif node_type == 'source':
      source_features = ['bias']
      selected_feature = source_features
      
  elif node_type == 'news_source':
      source_news_features = ['Analytic','insight','cause','discrep','tentat','certain','differ','affiliation','power','reward','risk','work','leisure',
                            'money','relig','Tone','affect','WC','WPS','num_nouns','num_propernouns','num_personalnouns','num_ppssessivenouns',
                            'num_whpronoun','num_determinants','num_whdeterminants','num_cnum','num_adverb','num_interjections','num_verb','num_adj',
                            'num_vbd','num_vbg','num_vbn','num_vbp','num_vbz','focuspast','focusfuture','i','we','you','shehe','quant','compare','Exclam',
                            'negate','swear','netspeak','interrog','count_uppercased','percentage_stopwords','AllPunc','Quote', 'lexical_diversity','wlen',
                            'gunning_fog_index','smog_index','flesch_kincaid_grade_level','Anger','Anticipation','Disgust','Fear','Joy', 
                                                    'Sadness', 'Surprise', 'Trust','neg','pos','posemo','negemo','anx']+['bias']
      selected_feature = source_news_features

  df = impute_values(df, selected_feature, impute)

  if scale:
    scaler = StandardScaler()
    df[selected_feature] = scaler.fit_transform(df[selected_feature])
  print("no of features for "+node_type,len(selected_feature))
  return df, selected_feature