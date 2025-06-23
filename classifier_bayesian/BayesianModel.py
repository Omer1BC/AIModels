import os
#Word to index mapping: {word:idx}
mapping = {}
#The each encoded word has pair (ham_count,spam_count)
Theta = []
def y(classifcation):
  if classifcation == "spam":
    return 1
  elif classifcation == "ham":
    return 0
  else:
    return -1
# Get lowercase alphanumeric words, or money
def clean(word):
  s = ""
  for c in word.lower():
    if c.isalnum() or c == "$" or c== ".":
      s += c
  return s
#Extracts the words from a given line
def get_clean_words(line):
  return [clean(word) for word in line.strip().split(" ") if clean(word)]
def compute_theta_and_mapping(ham_path,spam_path):
  HAM = 0
  SPAM = 0
  #Wakl each file starting at ham root and spam root
  for root,dirs,files in os.walk(spam_path):
    SPAM = len(files)
    for file in files:
      #A word has one contribution per file
      words_per_file = set()
      with open(os.path.join(root,file),'r') as f:
        for line in f:
          #pre process
          words = get_clean_words(line)
          for word in words:
            #Skip already counted words
            if word in words_per_file:
              continue
            words_per_file.add(word)
            ''' Each word is encoded as the next available index'''
            if word not in mapping:
              mapping[word] = (len(Theta))
              Theta.append([0,0])

            Theta[mapping[word]][y("spam")] += 1 #Update the count
  #Same prcoess for for ham
  for root,dirs,files in os.walk(ham_path):
    HAM = len(files)
    for file in files:
      words_per_file = set()
      with open(os.path.join(root,file),'r') as f:
        for line in f:
          words = get_clean_words(line)
          for word in words:
            if word in words_per_file:
              continue
            words_per_file.add(word)
            if word not in mapping:
              mapping[word] = (len(Theta))
              Theta.append([1,0])
            else:
              if Theta[mapping[word]][y("ham")] == 0:
                Theta[mapping[word]][y("ham")] = 1
            Theta[mapping[word]][y("ham")] += 1
  return SPAM,HAM
#Ecnode a test file
def encode(filename,n):
  X = [0]*n
  #Go through every line in the file
  with open(filename,'r') as f:
    for line in f:
      words = get_clean_words(line)
      #Use the mapping to find the index
      for word in words:
        if word not in mapping: #Skip words not in the vocab
          continue
        X[mapping[word]] = 1
    return X
#Find the liklihood from encoding X and classification Y
def likelyhood(X,y):
  res = 1
  for i in range(len(X)):
    proba = Theta[i][y]
    #For excluded words, take 1 - probability
    res *= proba if X[i] == 1 else 1 - proba
  return res
#Classify each test file
def test(test_path,spam,ham):
  res = {}
  #Walk every test file
  for root,_,files in os.walk(test_path):
    for file in files:
      #Generate encoding
      X = encode(os.path.join(root,file),len(Theta))
      #Calculate probabilites following the baye's formula
      P_SPAM = (spam/(ham+spam))
      P_HAM = (ham/(ham+spam))
      numerator = likelyhood(X,1)*P_SPAM
      denominator = likelyhood(X,1)*P_SPAM + likelyhood(X,0)*P_HAM
      guess = numerator/denominator
      #Set the guess based on a .5 threshold
      res[file] = ["SPAM" if guess >= 0.5 else "HAM",guess]
  return res

ROOT = "."
HAM_PATH = f"{ROOT}/ham"
SPAM_PATH = f"{ROOT}/spam"
TEST_PATH = f"{ROOT}/test"
SPAM,HAM = compute_theta_and_mapping(HAM_PATH,SPAM_PATH)
#Calculate theta_j with the ham and spam counts for each encoded word
Theta = [[(ham_c + 1)/(HAM + 2),(spam_c + 1)/(SPAM + 2)] for ham_c, spam_c in Theta ]
res = test(TEST_PATH,SPAM,HAM)
# Q1
# 1.1
print(f'Word Mapping has {len(mapping)} Words:\n',mapping)
#1.2 - see compute_theta_and_mapping() for the encoding strategy
#1.3 - see liklihood()
#1.4 - see test()
#1.5
print('Results:\n',res)