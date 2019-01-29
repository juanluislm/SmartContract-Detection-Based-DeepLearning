function Deposit() public payable
if(msg.value >= MinDeposit)
balances[msg.sender]+=msg.value;
TransferLog.AddMessage(msg.sender,msg.value,"Deposit");
function CashOut(uint _am)
if(_am<=balances[msg.sender])
if(msg.sender.call.value(_am)())
balances[msg.sender]-=_am;
TransferLog.AddMessage(msg.sender,_am,"CashOut");
