function score = pcp(err,l,p)
score = (err(1,:) <= l*p)&(err(2,:) <= l*p);

