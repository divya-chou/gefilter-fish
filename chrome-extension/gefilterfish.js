// Grab the ASIN and product title
var asin         = document.getElementById("ASIN").value;
var productTitle = document.getElementById("productTitle").innerText;
console.log("doing something");

// Check if ASIN is in our database
$.get( "https://gefilterfish.science/in_db", { asin: asin })
  .done( function ( in_db ) {

    // If in the DB then modify the page
    if (in_db) {

      // Grab the topics and add in the buttons
      $.get( "https://gefilterfish.science/model", { asin: asin })
        .done( function( data ) { 

        // Replace the title of the section
        $('h3[data-hook="lighthut-title"]').replaceWith('<h3 data-hook="lighthut-title" class="a-spacing-base">Topics in customer reviews</h3>');
     
          // Add in buttons
          console.log("got here");
          $('.cr-lighthouse-terms').replaceWith('<div class="cr-lighthouse-terms">');
          for (var i = 0; i < data.topic.length; i++) {
            $('.cr-lighthouse-terms').append('<form action="https://gefilterfish.science/reviews?topic='+i+'&title='+encodeURIComponent(productTitle)+'" method="post">'+
                                               ' <button class="cr-lighthouse-term" name="topic'+i+' type="submit">'+data.topic[i]+'</button>'+
                                             '</form>');
          }
          $('.cr-lighthouse-terms').append('</div>');
          $('form').css({"display": "inline"});
          $('.cr-lighthouse-terms').css({"max-height": "200px"});
      
        });

    } 

  });




