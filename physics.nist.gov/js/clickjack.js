    if (self === top) {
      var antiClickjack = document.getElementById("antiClickjack");
      if ( (antiClickjack != null) && (antiClickjack.parentNode != null) ) {
        antiClickjack.parentNode.removeChild(antiClickjack);
      }
    } else {
      top.location = self.location;
    }
