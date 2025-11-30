/*
From a given person, find all accounts they own. Then locate other accounts that sent money to 
those accounts through 1–3 transfers whose timestamps are in strictly descending order, with the 
first transfer before a given time and the last after another time.

For each such “other” account, find loans that deposited money into it within a specified time window. 
Sum the loan amounts and balances for each account, round them to three decimals, and return the 
account ID along with these totals. Sort by total loan amount (descending) and then by account ID 
(ascending).
*/
MATCH (p:Person {id:%d})-[e1:own]->(acc:Account) <-[e2:transfer*1..3]-(other:Account) 
WHERE isDesc(getMemberProp(e2, 'timestamp'))=true 
  AND head(getMemberProp(e2, 'timestamp')) < %d 
  AND last(getMemberProp(e2, 'timestamp')) > %d 
WITH DISTINCT other 
MATCH (other)<-[e3:deposit]-(loan:Loan) 
WHERE e3.timestamp > %d 
  AND e3.timestamp < %d 
WITH DISTINCT other.id AS otherId, loan.loanAmount AS loanAmount, loan.balance AS loanBalance 
WITH otherId AS otherId, sum(loanAmount) as sumLoanAmount, sum(loanBalance) as sumLoanBalance 
RETURN otherId, round(sumLoanAmount * 1000) / 1000 as sumLoanAmount, round(sumLoanBalance * 1000) / 1000 as sumLoanBalance 
ORDER BY sumLoanAmount DESC, otherId ASC;