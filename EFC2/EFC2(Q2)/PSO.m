%-----------------------------------------------------------------------------%
%Author: André Barros de Medeiros
%Date: 11/23/2019
%Copyright: free to use, copy, and modify
%Description:  PID Control via evolutionary algorithm which operates as a 
%              continuous space optimization metaheuristic
%Important: if user is working with Octave instead of Matlab, the command
%              "stepinfo" should be replaced by "stepinfo_Q1"
%Note: It is not a garanteed that fitness will reach 1 on first try. In
%               that case, run again if needed. For further information 
%               refer to PDF in repository: "PDFs/EFC2_EA072_2s2019"
%      For example of graphs of a successfull (final fitness=1) evolution,
%               see repository: "EFC2/EFC2(Q2)/Graphs"
%Pseudo-Code: (Global Best)
%   
%     FOR each particle i
%         FOR each dimension d
%             Initialize position Xid randomly within permissible range
%             Initialize velocity Vid randomly within permissible range
%         END
%     END
%     WHILE maximum iterations or minimum error criteria not reached
%         iteration k
%         FOR each particle i
%             Calculate Fitness
%             IF fitness is better than own best
%                 update own best (Pid)
%             END
%         END
%         Choose the particle with the best fitness value (Pgd)
%         FOR each Particle i
%             FOR each dimension d
%                 Calculate velocity according to the equation: Vid(k+1) = W*Vid(k)+C1*rand(Pid-Xid)+C1*rand(Pgd-Xid)
%                 Update particle position according to equation: Xid(k+1) = Xid(k)+Vid(k+1)
%             END
%         END
%         Add to iteration
%     END
%-----------------------------------------------------------------------------%


