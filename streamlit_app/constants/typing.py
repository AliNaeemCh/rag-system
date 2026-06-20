TYPING_HTML = """
<div class="typing">
  <span></span>
  <span></span>
  <span></span>
</div>

<style>

.typing {
  display:flex;
  align-items:center;
  gap:6px;
}

.typing span {

  width:7px;
  height:7px;
  border-radius:50%;
  background:#999;

  animation:typing 1.6s infinite ease-in-out;
}


.typing span:nth-child(2){
  animation-delay:.2s;
}


.typing span:nth-child(3){
  animation-delay:.4s;
}


@keyframes typing {

0%,60%,100%{
transform:translateY(0);
opacity:.45;
}

30%{
transform:translateY(-4px);
opacity:1;
}

}

</style>
"""