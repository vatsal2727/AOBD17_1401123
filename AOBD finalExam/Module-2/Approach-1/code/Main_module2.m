clear all;
clc
close all;
tic

 Y = [1 0 0 0 0 1 0 1 1 1;0 1 0 0 0 1 1 1 0 1;0 0 1 0 0 0 1 1 1 0;
     0 0 0 1 0 1 0 1 1 0;0 0 0 0 1 1 0 1 1 1;0 1 0 0 0 1 1 1 1 1;
     1 0 0 0 0 1 0 1 1 1;0 0 0 0 0 1 1 1 1 0;0 0 0 0 0 0 0 1 1 1;
     0 0 0 0 0 1 0 1 1 0];

R=Y;

imagesc(Y);
ylabel('Skill');
xlabel('Users');

X = [3;2.5;3;4;3;2;3;2;3;2]
X_max=max(X)+0.2;
X = X./X_max;

Theta = [2;3;2;3.6;4;3.7;3;2;2.7;2.9]
Theta_max= max(Theta)+0.2;
Theta=Theta./Theta_max;

num_users = 10; num_skills = 10; num_features = 1;    
J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_skills, ...
               num_features, 1.5);
checkCostFunction(1.5);

skillList = loadSkillsList();
my_skills = zeros(10, 1);

my_skills(4)=1;
my_skills(7)=1;

fprintf('\n\nNew user skills:\n');
for i = 1:length(my_skills)
    if my_skills(i) > 0 
        user_skill_minis=skillList{i};
        fprintf('User skill %s\n',skillList{i});
    end
end

% add new users skill
Y = [Y my_skills];
R = [R (my_skills ~= 0)];

%  Normalize skills
[Ynorm, Ymean] = normalizeRatings(Y, R);
%  Useful Values
num_users = size(Y, 2);
num_skills = size(Y, 1);
num_features = 1;

% Set Initial Parameters (Theta, X)
Theta(num_users,num_features) = randi([1 3],1, num_features)/Theta_max;

initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
lambda = 0.1;
theta = fmincg (@(t)(cofiCostFunc(t, Y, R, num_users, num_skills, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_skills*num_features), num_skills, num_features);
Theta = reshape(theta(num_skills*num_features+1:end), ...
                num_users, num_features);
            
fprintf('Recommender system learning completed.\n');

p = X * Theta';

figure;
plot(X,p)
hold on;
plot(X,Y,'o')
hold off;

skillList = loadSkillsList();

threshold=0.3;

[r, ix] = sort(p(:,num_users), 'descend');

fprintf('\nTop skill recommendations for you:\n');

for k=1:10
        j = ix(k);
        fprintf('Predicting rating new skill for future : %s\n', skillList{j});
end

toc