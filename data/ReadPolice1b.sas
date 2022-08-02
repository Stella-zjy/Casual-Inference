*** edited to read and describe police data -- JDA 3/21/05 ***;
*** artifical dep variables used to reconstruct ***;
*** first-stage uses coding that matches sherman-berk 84, while RF/OLS uses results from 1988 JASA paper ***;

/*                                                                             
                    SAS CONTROL STATEMENTS FOR ICPSR 8250                      
          SPECIFIC DETERRENT EFFECTS OF ARREST FOR DOMESTIC ASSAULT:           
                   MINNEAPOLIS, 1981-1982 [POLICE DATA SET]                    
                           (SECOND ICPSR EDITION)                              
                               October, 1993                                   
                                                                               
   Five SAS setup sections are provided in this file for the LRECL             
   version of this data collection.  These sections are listed below;          
                                                                               
   PROC FORMAT assigns value labels (SAS formats) to all variables which       
   have value descriptions in the codebook. Some value labels were             
   abbreviated and truncated in order to meet the SAS limitation on the        
   number of characters allowed for each value label.                          
                                                                               
   INPUT contains the SAS statements which assign the variable names and       
   specify the beginning and ending column locations in the LRECL data         
   file for each variable.                                                     
                                                                               
   VARIABLE LABELS assigns variable labels for all variables in the data       
   file.                                                                       
                                                                               
   FORMATS associates the SAS formats defined by the PROC FORMAT step          
   with the variables named in the INPUT statement.                            
                                                                               
   MISSING VALUE RECODES contains SAS program statements which convert         
   missing values to the SAS missing value ".".                                
                                                                               
   Users may combine and modify these sections or parts of these sections      
   to suit their specific needs.  Users will also need to change the           
   file-specification in the INFILE statement to an appropriate filename       
   for their system.                                                           
                                                                               
*/                                                                             
                                                                               
                                                                               
/* SAS PROC FORMAT FOR 8250,                                                   
   SPECIFIC DETERRENT EFFECTS OF ARREST FOR DOMESTIC ASSAULT: MINNEAPOLIS,     
   1981-1982 (POLICE DATA FILE)  */                                            


libname savefmts 'C:\D_drive\PROJECTS\IVtalk\Sherman_Berk\mySASfmts';                                                                   
PROC FORMAT lib=savefmts;                                                                   
VALUE V2FMT                                                                    
     1 ="pink"                                                                 
     2 ="yellow"                                                               
     3 ="blue"                                                                 
     ;                                                                         
VALUE V7FMT                                                                    
     0 ="no"                                                                   
     1 ="yes"                                                                  
     ;                                                                         
VALUE V9FMT                                                                    
     1 ="White"                                                                
     2 ="Black"                                                                
     3 ="Indian"                                                               
     4 ="Asian"                                                                
     5 ="Hispanic"                                                             
     6 ="1 and 2, 3, 4, and/or 5"                                              
     7 ="2, 3, 4, and/or 5"                                                    
     ;                                                                         
VALUE V10FMT                                                                   
     1 ="White"                                                                
     2 ="Black"                                                                
     3 ="Indian"                                                               
     4 ="Asian"                                                                
     5 ="Hispanic"                                                             
     6 ="1 and 2, 3, 4, and/or 5"                                              
     7 ="2, 3, 4, and/or 5"                                                    
     9 ="unknown"                                                              
     ;                                                                         
VALUE V11FMT                                                                   
     0 ="no"                                                                   
     1 ="yes"                                                                  
     9 ="unknown"                                                              
     ;                                                                         
VALUE V12FMT                                                                   
     0 ="no"                                                                   
     1 ="yes"                                                                  
     9 ="unknown"                                                              
     ;                                                                         
VALUE V13FMT                                                                   
     1 ="polite"                                                               
     2 ="rude"                                                                 
     9 ="unknown"                                                              
     ;                                                                         
VALUE V14FMT                                                                   
     1 ="polite"                                                               
     2 ="rude"                                                                 
     9 ="unknown"                                                              
     ;                                                                         
VALUE V15FMT                                                                   
     1 ="blank"                                                                
     2 ="no weapons"                                                           
     3 ="body"                                                                 
     4 ="blunt object"                                                         
     5 ="sharp object"                                                         
     6 ="other"                                                                
     7 ="3, 4, and 5"                                                          
     8 ="4 and 5"                                                              
     9 ="unknown"                                                              
     ;                                                                         
VALUE V16FMT                                                                   
     1 ="no guns"                                                              
     2 ="guns removed"                                                         
     3 ="guns left in house"                                                   
     9 ="unknown"                                                              
     ;                                                                         
VALUE V17FMT                                                                   
     0 ="no"                                                                   
     1 ="yes"                                                                  
     9 ="unknown"                                                              
     ;                                                                         
VALUE V18FMT                                                                   
     1 ="no assault"                                                           
     2 ="assaulted before"                                                     
     3 ="assaulted after"                                                      
     9 ="unknown"                                                              
     ;                                                                         
VALUE V19FMT                                                                   
     0 ="no"                                                                   
     1 ="yes"                                                                  
     9 ="other treatment"                                                      
     99 ="unknown"                                                             
     ;                                                                         
VALUE V20FMT                                                                   
     0 ="no"                                                                   
     1 ="yes"                                                                  
     9 ="other treatment"                                                      
     99 ="unknown"                                                             
     ;                                                                         
VALUE V21FMT                                                                   
     0 ="no"                                                                   
     1 ="yes"                                                                  
     9 ="other treatment"                                                      
     99 ="unknown"                                                             
     ;                                                                         
VALUE V22FMT                                                                   
     1 ="arrest"                                                               
     2 ="advise"                                                               
     3 ="suspect told to leave"                                                
     4 ="other"                                                                
     ;                                                                         
VALUE V23FMT                                                                   
     1 ="blank"                                                                
     2 ="more than one party injured"                                          
     3 ="identity of assailant is unclear"                                     
     4 ="party assaults police officer"                                        
     5 ="victim makes citizen's arrest"                                        
     6 ="victim's injury may pose threat to life"                              
     7 ="injury constitutes an aggravated assault"                             
     8 ="victim has order of protection against/s"                             
     9 ="other"                                                                
     99 ="unknown"                                                             
     ;                                                                         
VALUE V24FMT                                                                   
     1 ="blank"                                                                
     2 ="more than one party injured"                                          
     3 ="identity of assailant is unclear"                                     
     4 ="party assaults police officer"                                        
     5 ="victim makes citizen's arrest"                                        
     6 ="victim's injury may pose threat to life"                              
     7 ="injury constitutes an aggravated assault"                             
     8 ="victim has order of protection against/s"                             
     9 ="other"                                                                
     99 ="unknown"                                                             
     ;                                                                         
VALUE V25FMT                                                                   
     1 ="blank"                                                                
     2 ="more than one party injured"                                          
     3 ="identity of assailant is unclear"                                     
     4 ="party assaults police officer"                                        
     5 ="victim makes citizen's arrest"                                        
     6 ="victim's injury may pose threat to life"                              
     7 ="injury constitutes an aggravated assault"                             
     8 ="victim has order of protection against/s"                             
     9 ="other"                                                                
     99 ="unknown"                                                             
     ;                                                                         
VALUE V26FMT                                                                   
     1 ="blank"                                                                
     2 ="more than one party injured"                                          
     3 ="identity of assailant is unclear"                                     
     4 ="party assaults police officer"                                        
     5 ="victim makes citizen's arrest"                                        
     6 ="victim's injury may pose threat to life"                              
     7 ="injury constitutes an aggravated assault"                             
     8 ="victim has order of protection against/s"                             
     9 ="other"                                                                
     99 ="unknown"                                                             
     ;                                                                         
VALUE V27FMT                                                                   
     1 ="husband"                                                              
     2 ="male, unidentified"                                                   
     3 ="woman, unidentified"                                                  
     4 ="separated or divorced, husband"                                       
     5 ="boyfriend or lover"                                                   
     6 ="other"                                                                
     7 ="wife, girlfriend, or lover"                                           
     9 ="unknown"                                                              
     ;                                                                         
                                                                               
/* SAS INPUT FOR 8250,                                                         
   SPECIFIC DETERRENT EFFECTS OF ARREST FOR DOMESTIC ASSAULT: MINNEAPOLIS,     
   1981-1982 (POLICE DATA FILE) */                                             
  
filename inpolice 'C:\D_drive\PROJECTS\IVtalk\Sherman_Berk\5075022\ICPSR_08250\DS0001_Police_Data\file1.txt';
libname saveberk 'C:\D_drive\PROJECTS\IVtalk\Sherman_Berk\mySASds';


options fmtsearch=(savefmts);

DATA saveberk.police;                                                                          
INFILE inpolice LRECL=59;                                                      
INPUT                                                                          
  ID 1-9                                                                       
  T_RANDOM 10-10                                                               
  MONTH 11-12                                                                  
  YEAR 13-14                                                                   
  CLOCK 15-18                                                                  
  TIME 19-21                                                                   
  P_REPORT 22-22                                                               
  CCN $ 23-33                                                                  
  V_RACE 34-34                                                                 
  S_RACE 35-35                                                                 
  V_CHEM 36-36                                                                 
  S_CHEM 37-37                                                                 
  S_DMNOR1 38-38                                                               
  S_DMNOR2 39-39                                                               
  WEAPON 40-40                                                                 
  GUNS 41-41                                                                   
  P_ASSLT1 42-42                                                               
  P_ASSLT2 43-43                                                               
  P_CALMED 44-45                                                               
  P_REFER 46-47                                                                
  P_ADVISE 48-49                                                               
  T_FINAL 50-50                                                                
  REASON1 51-52                                                                
  REASON2 53-54                                                                
  REASON3 55-56                                                                
  REASON4 57-58                                                                
  R_RELATE 59-59                                                               
  ;                                                                            
                                                                               
/* SAS LABEL FOR 8250,                                                         
   SPECIFIC DETERRENT EFFECTS OF ARREST FOR DOMESTIC ASSAULT: MINNEAPOLIS,     
   1981-1982 (POLICE DATA FILE)  */                                            
                                                                               
LABEL                                                                          
  ID="ID Number"                                                               
  T_RANDOM="Police Sheet Color"                                                
  MONTH="Calendar Month"                                                       
  YEAR="Year"                                                                  
  CLOCK="Time of Incident"                                                     
  TIME="Time Spent at Scene by Police"                                         
  P_REPORT="Offense Report Made?"                                              
  CCN="Case Number"                                                            
  V_RACE="Victim's Ethnicity"                                                  
  S_RACE="Suspect's Ethnicity"                                                 
  V_CHEM="Victim Under Chemical Influence?"                                    
  S_CHEM="Suspect Under Chemical Influence?"                                   
  S_DMNOR1="Suspect's Demeanor on Arrival"                                     
  S_DMNOR2="Suspect's Demeanor on Departure"                                   
  WEAPON="Weapons Used in Assault"                                             
  GUNS="Disposition of Guns"                                                   
  P_ASSLT1="Police Officer Assaulted?"                                         
  P_ASSLT2="Assaulted Before or After?"                                        
  P_CALMED="Officers Calmed Things Down"                                       
  P_REFER="Officers Referred Parties"                                          
  P_ADVISE="Officers Advised Parties"                                          
  T_FINAL="Final Disposition"                                                  
  REASON1="First Reason"                                                       
  REASON2="Second Reason"                                                      
  REASON3="Third Reason"                                                       
  REASON4="Fourth Reason"                                                      
  R_RELATE="Relationship of Suspect to Vict."                                  
  ;                                                                            
                                                                               
/* SAS FORMAT FOR 8250,                                                        
   SPECIFIC DETERRENT EFFECTS OF ARREST FOR DOMESTIC ASSAULT: MINNEAPOLIS,     
   1981-1982 (POLICE DATA FILE)  */                                            
                                                                               
FORMAT                                                                         
     T_RANDOM  V2FMT.                                                          
     P_REPORT  V7FMT.                                                          
     V_RACE    V9FMT.                                                          
     S_RACE    V10FMT.                                                         
     V_CHEM    V11FMT.                                                         
     S_CHEM    V12FMT.                                                         
     S_DMNOR1  V13FMT.                                                         
     S_DMNOR2  V14FMT.                                                         
     WEAPON    V15FMT.                                                         
     GUNS      V16FMT.                                                         
     P_ASSLT1  V17FMT.                                                         
     P_ASSLT2  V18FMT.                                                         
     P_CALMED  V19FMT.                                                         
     P_REFER   V20FMT.                                                         
     P_ADVISE  V21FMT.                                                         
     T_FINAL   V22FMT.                                                         
     REASON1   V23FMT.                                                         
     REASON2   V24FMT.                                                         
     REASON3   V25FMT.                                                         
     REASON4   V26FMT.                                                         
     R_RELATE  V27FMT.                                                         
     ;                                                                         
                                                                               
/* SAS MISSING VALUE RECODE FOR 8250,                                          
   SPECIFIC DETERRENT EFFECTS OF ARREST FOR DOMESTIC ASSAULT: MINNEAPOLIS,     
   1981-1982 (POLICE DATA FILE)  */                                            
                                                                               
  IF MONTH=99 THEN MONTH=.;                                                    
  IF YEAR=99 THEN YEAR=.;                                                      
  IF CLOCK=9999 THEN CLOCK=.;                                                  
  IF TIME=999 THEN TIME=.;                                                     
  IF S_RACE=9 THEN S_RACE=.;                                                   
  IF V_CHEM=9 THEN V_CHEM=.;                                                   
  IF S_CHEM=9 THEN S_CHEM=.;                                                   
  IF S_DMNOR1=9 THEN S_DMNOR1=.;                                               
  IF S_DMNOR2=9 THEN S_DMNOR2=.;                                               
  IF WEAPON=9 THEN WEAPON=.;                                                   
  IF GUNS=9 THEN GUNS=.;                                                       
  IF P_ASSLT1=9 THEN P_ASSLT1=.;                                               
  IF P_ASSLT2=9 THEN P_ASSLT2=.;                                               
  IF P_CALMED=99 THEN P_CALMED=.;                                              
  IF P_REFER=99 THEN P_REFER=.;                                                
  IF P_ADVISE=99 THEN P_ADVISE=.;                                              
  IF REASON1=99 THEN REASON1=.;                                                
  IF REASON2=99 THEN REASON2=.;                                                
  IF REASON3=99 THEN REASON3=.;                                                
  IF REASON4=99 THEN REASON4=.;                                                
  IF R_RELATE=9 THEN R_RELATE=.;                                               
                                                                               
proc contents;
proc means;
proc freq;
 *where t_final ne 4;
 tables t_random*t_final v_race month r_relate;

 data two;
   set saveberk.police;

   z_arrest=(t_random=1);
   z_advise=(t_random=2);
   z_separate=(t_random=3);

   d_arrest=(t_final=1);
   d_advise=(t_final=2);
   d_separate=(t_final=3);
   d_other=(t_final=4);

   z_coddled=(z_arrest=0);
   d_coddled=(d_arrest=0);

   y82=(year=82);
   q1=(1<=month<=3); q2=(4<=month<=6); q3=(7<=month<=9);
   black=(v_race=2); native=(v_race=3); other_nw=(4<=v_race<=5); mixed=(v_race ne s_race);
   s_influence=(s_chem=1);
   gun_used=(2<=guns<=3);
   o_weapon=(4<=weapon<=5);
   anyweapon=(gun_used=1 or o_weapon=1);
   pol_asslt=(p_asslt2=2);
   husband=(r_relate=1); boyfriend=(r_relate=5);
   nonwhite=(v_race ne 1);
   
proc means data=two;
 *where t_final ne 4;
 var z_arrest z_advise z_separate d_arrest d_advise d_separate d_other 
     z_coddled d_coddled y82 q1-q3 black native other_nw mixed s_influence 
     gun_used o_weapon anyweapon pol_asslt;
 

proc reg data=two;
title 'single-dummy first stages';
 where t_final ne 4;
  model d_coddled=z_coddled;
  model d_coddled=z_coddled y82 q1-q3;
  model d_coddled=z_coddled y82 q1-q3 black native other_nw mixed ; 
  model d_coddled=z_coddled y82 q1-q3 black native other_nw mixed anyweapon s_influence; 

  model d_coddled=z_coddled y82 q1-q3 nonwhite mixed; 
  model d_coddled=z_coddled y82 q1-q3 nonwhite mixed anyweapon s_influence; 


proc reg data=two;
title 'two-dummy first stages (two types of coddling)';
 where t_final ne 4;
  eq1: model d_advise=z_advise z_separate;
  eq2: model d_separate=z_advise z_separate;

  eq1: model d_advise=z_advise z_separate y82 q1-q3;
  eq2: model d_separate=z_advise z_separate y82 q1-q3;

  eq1: model d_advise=z_advise z_separate y82 q1-q3 nonwhite mixed anyweapon s_influence;
  eq2: model d_separate=z_advise z_separate y82 q1-q3 nonwhite mixed anyweapon s_influence;

*** PART II: SIMULATE OUTCOMES [USE PROC RANKS, WITH RANK BY ASSIGNMENT!] ***;

  data three;
  set two;

  /* from Table 4; Berk and Sherman, 1988 */
  pz_separ=1/(1+exp(1.21)); pz_arrest=1/(1+exp(1.21+.9)); pz_advise=1/(1+exp(1.21+.21));

  /* from Table 6; Berk and Sherman, 1988 */
  pd_separ=1/(1+exp(1.05)); pd_arrest=1/(1+exp(1.05+.82)); pd_advise=1/(1+exp(1.05+.46));

  if t_final ne 4;

  data three; set three;
  order=_N_;

proc sort data=three; by t_random;

proc rank data=three out=four_z fraction;
 var order;
 ranks z_rank;
 by t_random;

data five_z; set four_z;
 fail_z=0;
 if ((z_arrest eq 1) and (z_rank lt pz_arrest)) then fail_z=1;
 if ((z_advise eq 1) and (z_rank lt pz_advise)) then fail_z=1;
 if ((z_separate eq 1) and (z_rank lt pz_separ)) then fail_z=1;

proc reg data=five_z;
title "reconstructed RF";
model fail_z = z_coddled;
model fail_z = z_advise z_separate;
model fail_z = z_coddled y82 q1-q3 nonwhite mixed anyweapon s_influence;
model fail_z = z_advise z_separate y82 q1-q3 nonwhite mixed anyweapon s_influence;
test z_advise=z_separate;

proc sort data=three; by t_final;

proc rank data=three out=four_d fraction;
 var order;
 ranks d_rank;
 by t_final;

data five_d; set four_d;
 fail_d=0;
 if ((d_arrest eq 1) and (d_rank lt pd_arrest)) then fail_d=1;
 if ((d_advise eq 1) and (d_rank lt pd_advise)) then fail_d=1;
 if ((d_separate eq 1) and (d_rank lt pd_separ)) then fail_d=1;

proc reg data=five_d;
title "reconstructed OLS";
model fail_d = d_coddled;
model fail_d = d_advise d_separate;
test d_advise=d_separate;
model fail_d = d_coddled  y82 q1-q3 nonwhite mixed anyweapon s_influence;
model fail_d = d_advise d_separate  y82 q1-q3 nonwhite mixed anyweapon s_influence;
test d_advise=d_separate;

*** 2SLS sequence: use simulated fail_z ***;

proc syslin data=five_z 2sls;
title "2SLS -- single dummy -- no covariates";
endogenous fail_z d_coddled;
instruments z_coddled;
model fail_z = d_coddled;

proc syslin data=five_z 2sls;
title "2SLS -- two dummies -- no covariates";
endogenous fail_z d_advise d_separate;
instruments z_advise z_separate;
model fail_z = d_advise d_separate;
test d_advise=d_separate;

proc syslin data=five_z 2sls;
title "2SLS -- single dummy -- with covariates";
endogenous fail_z d_coddled;
instruments z_coddled y82 q1-q3 nonwhite mixed anyweapon s_influence;
model fail_z = d_coddled y82 q1-q3 nonwhite mixed anyweapon s_influence;

proc syslin data=five_z 2sls;
title "2SLS -- two dummies -- with covariates";
endogenous fail_z d_advise d_separate;
instruments z_advise z_separate y82 q1-q3 nonwhite mixed anyweapon s_influence;
model fail_z = d_advise d_separate y82 q1-q3 nonwhite mixed anyweapon s_influence;
test d_advise=d_separate;

run;
