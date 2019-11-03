%-----------------------------------------------------------------------------%
%Author: André Barros de Medeiros
%Date: 11/03/2019
%Copyright: free to use, copy, and modify
%Description:  PID Control via evolutionary algorithm which operates as a 
%              continuous space optimization metaheuristic
%Important: if user is working with Octave instead of Matlab, the command
%              "stepinfo" should be replaced by "stepinfo_Q1"
%Note: It is not a garanteed that fitness will reach 1 on first try. In
%               that case, run again if needed. For further information 
%               refer to PDF in repository: "PDFs/EFC2_EA072_2s2019"
%      For example of graphs of a successfull (final fitness=1) evolution,
%               see repository: "EFC2/EFC2(Q1)/Graphs"
%-----------------------------------------------------------------------------%

clear all;
close all;
% Define a Population size [tam_pop], with an even number
tam_pop = 200;
% Define the crossover and mutation rates
pc = 0.5;
pm = 0.7;
% Define the variable excursion interval
v_inf = 0.0;
v_sup = 20.0;
% Define number of generations
n_ger = 50;
% Vectors for maintaining best individual and its fitness every generation
v_melhor_fitness = [];
v_fitness_medio = [];
v_melhor_ind = [];
% Generates initial population in defined variable excursion interval. 
%       Each individaual is represented in a line of the matrix [pop].
pop = v_inf+(v_sup-v_inf)*rand(tam_pop,3);
% Evaluate initial population
for i=1:tam_pop,
    kp = pop(i,1);
    kd = pop(i,2);
    ki = pop(i,3);
    S = stepinfo(G_MF(kp,kd,ki));
    [Gm,Pm,Wcg,Wcp] = margin(G_MF(kp,kd,ki));
    t1 = 0.0;t2 = 0.0;t3 = 0.0;
    if isnan(S.SettlingTime) | isnan(S.RiseTime) | Pm <= 0,
        fitness(i,1) = 0;;
    else
        if S.SettlingTime > 0.5,
            t1 = S.SettlingTime - 0.5;
        end
        if S.RiseTime > 0.04,
            t2 = S.RiseTime - 0.04;
        end
        t3 = (Pm-60)^2;
        fitness(i,1) = 1/(t1+(0.5/0.04)*t2+t3+1);
    end
end
% Generate next population loop (until max generations reached)
[melhor_fitness,melhor_ind] = max(fitness);
v_melhor_fitness = [v_melhor_fitness;melhor_fitness];
v_fitness_medio = [v_fitness_medio;mean(fitness)];
v_melhor_ind = [v_melhor_ind;pop(melhor_ind,:)];
v_delta = [];
for k = 1:n_ger,
    % Instead of using the Roulette Wheel Algorithm, we use a 3-sized Tournament Algorithm, 
    %   because it is better to eliminate the Individuals with fitness=0 to improve performance
    candidatos = [];
    for i=1:tam_pop,
        if fitness(i,1) > 0,
            candidatos = [candidatos;i];
        end
    end
    n_tam_pop = length(candidatos);
    n_pop = [];
%    [[1:tam_pop]' pop fitness]
    for i=1:tam_pop,
        v_aux = randperm(n_tam_pop)';
        torneio = v_aux(1:3,1);
        c_fitness(1,1) = fitness(candidatos(torneio(1,1),1),1);
        c_fitness(2,1) = fitness(candidatos(torneio(2,1),1),1);
        c_fitness(3,1) = fitness(candidatos(torneio(3,1),1),1);
        [v_max,ind_max] = max(c_fitness);
        n_pop = [n_pop;pop(candidatos(torneio(ind_max,1),1),:)];
    end
    % Apply crossover 
    for j=1:(tam_pop/2),
        if rand(1,1) <= pc,
            % 50% chance of Arithmetic Crossover
            if rand(1,1) <= 0.5,
                a = -0.1+1.2*rand(1,1);
                n_pop1 = a*n_pop(2*(j-1)+1,:)+(1-a)*n_pop(2*(j-1)+2,:);
                n_pop2 = (1-a)*n_pop(2*(j-1)+1,:)+a*n_pop(2*(j-1)+2,:);
                n_pop(2*(j-1)+1,:) = n_pop1;
                n_pop(2*(j-1)+2,:) = n_pop2;
            else
                % 50% chance of Uniform Crossover
                for z=1:3,
                    if rand(1,1) <= 0.5;
                        n_pop1(1,z) = n_pop(2*(j-1)+1,z);
                        n_pop2(1,z) = n_pop(2*(j-1)+2,z);
                    else
                        n_pop1(1,z) = n_pop(2*(j-1)+2,z);
                        n_pop2(1,z) = n_pop(2*(j-1)+1,z);
                    end
                end
                n_pop(2*(j-1)+1,:) = n_pop1;
                n_pop(2*(j-1)+2,:) = n_pop2;
            end
        end
    end
    % Apply non-uniform mutation
    n_mut = round(tam_pop*3*pm);
    v_aux = randperm(tam_pop*3)';
    bits_mutados = v_aux(1:n_mut,1);
    for i=1:n_mut,
        if rem(bits_mutados(i),3) == 0,
            linha = fix(bits_mutados(i)/3);
            coluna = 3;
        else
            linha = fix(bits_mutados(i)/3)+1;
            coluna = rem(bits_mutados(i),3);
        end
        if rand(1,1) <= 0.5,
            delta = mut_nunif(k,v_sup-n_pop(linha,coluna),n_ger);
            v_delta = [v_delta;delta];
            n_pop(linha,coluna) = n_pop(linha,coluna) + delta;
        else
            delta = mut_nunif(k,n_pop(linha,coluna)-v_inf,n_ger);
            v_delta = [v_delta;delta];
            n_pop(linha,coluna) = n_pop(linha,coluna) - delta;
        end
    end
    % Evaluate new population
    for i=1:tam_pop,
        kp = n_pop(i,1);
        kd = n_pop(i,2);
        ki = n_pop(i,3);
        S = stepinfo(G_MF(kp,kd,ki));
        [Gm,Pm,Wcg,Wcp] = margin(G_MF(kp,kd,ki));
        t1 = 0.0;t2 = 0.0;t3 = 0.0;
        if isnan(S.SettlingTime) | isnan(S.RiseTime) | Pm <= 0,
            fitness(i,1) = 0;
        else
            if S.SettlingTime > 0.5,
                t1 = S.SettlingTime - 0.5;
            end
            if S.RiseTime > 0.04,
                t2 = S.RiseTime - 0.04;
            end
            t3 = (Pm-60)^2;
            fitness(i,1) = 1/(t1+(0.5/0.04)*t2+t3+1);
        end
    end
    % Maintain best individual form previous generation if better than best from current one
    melhor_fitness1 = melhor_fitness;
    melhor_ind1 = melhor_ind;
    [melhor_fitness,melhor_ind] = max(fitness);
    if melhor_fitness1 > melhor_fitness,
        [pior_fitness,pior_ind] = min(fitness);
        n_pop(pior_ind,:) = pop(melhor_ind1,:);
        fitness(pior_ind,1) = melhor_fitness1;
        melhor_fitness = melhor_fitness1;
        melhor_ind = pior_ind;
    end
    pop = n_pop;
    v_melhor_fitness = [v_melhor_fitness;melhor_fitness];
    v_fitness_medio = [v_fitness_medio;mean(fitness)];
    v_melhor_ind = [v_melhor_ind;pop(melhor_ind,:)];
    kp = pop(melhor_ind,1);
    kd = pop(melhor_ind,2);
    ki = pop(melhor_ind,3);
    figure(1);step(G_MF(kp,kd,ki));drawnow;
    S = stepinfo(G_MF(kp,kd,ki));
    [Gm,Pm,Wcg,Wcp] = margin(G_MF(kp,kd,ki));
    disp(sprintf('Generation %d: T_sub = %g | T_acom = %g | Sobrs = %g | Margfase = %g',k,S.RiseTime,S.SettlingTime,S.Overshoot,Pm));
    disp(sprintf('Generation %d: The best values found were: k_p = %g; k_d = %g; k_i = %g',k,kp,kd,ki));
    disp(sprintf('Generation %d: This individuals fitness = %g',k,melhor_fitness));
end
figure(2);plot(v_melhor_fitness,'k');hold on;plot(v_fitness_medio,'r');hold off;
title('Best Fitness (black) and mean Fitness (red) of each Population throughout the Generations');
figure(3);plot(v_melhor_ind(:,1),'k');hold on;plot(v_melhor_ind(:,2),'r');plot(v_melhor_ind(:,3),'b');hold off;
title('Evolution of the PID control Gains: k_p (black) | k_d (red) | k_i (blue)');
xlabel('Number of Generations');
figure(4);plot(v_delta);
title('Non-Uniforrm Mutation Intensity throughout the Generations');
